import os
import time
import io
import bz2
import random
import json
import hashlib
import threading
import tkinter as tk
from email.utils import parsedate_to_datetime
import requests
import numpy
from shapely import geometry, ops
from xml.etree import ElementTree
from xml.sax.saxutils import quoteattr
import O4_UI_Utils as UI
import O4_File_Names as FNAMES
from O4_DSF_Budget import stable_id_key

overpass_servers = {
    "DE": "https://overpass-api.de/api/interpreter",
    "LZ": "https://lz4.overpass-api.de/api/interpreter",
    "FR": "https://api.openstreetmap.fr/oapi/interpreter",
    "KU": "https://overpass.kumi.systems/api/interpreter",
    "CH": "https://overpass.osm.ch/api/interpreter"
}
overpass_server_choice = "random"
osm_download_failure_policy = "abort"
OSM_FAILURE_POLICIES = ("abort", "continue_degraded", "prompt")
OSM_FAILED = 0
OSM_COMPLETE = 1
OSM_DEGRADED = 2
VALID_DATA = "VALID_DATA"
VALID_EMPTY = "VALID_EMPTY"
FAILED = "FAILED"
max_osm_tentatives = 3

# Regional public Overpass instances must not be treated as global fallbacks.
# The polygons are deliberately inset from the country borders.  A bbox which
# touches a border, crosses a border, or falls outside the known safe area is
# sent to a global instance instead of being sent to a regional instance which
# may return a misleading empty response.
_FRANCE_OVERPASS_COVERAGE = geometry.Polygon(
    [
        (-1.6, 48.5),
        (-0.8, 49.7),
        (1.5, 50.1),
        (4.2, 49.9),
        (6.2, 49.0),
        (6.4, 47.8),
        (6.3, 46.8),
        (5.9, 45.8),
        (5.2, 44.2),
        (4.3, 43.5),
        (2.0, 43.5),
        (0.2, 43.7),
        (-1.5, 43.5),
        (-1.6, 48.5),
    ]
)
_SWITZERLAND_OVERPASS_COVERAGE = geometry.Polygon(
    [
        (6.4, 47.0),
        (6.9, 47.55),
        (8.0, 47.7),
        (9.5, 47.7),
        (10.1, 47.3),
        (9.8, 46.4),
        (9.3, 46.0),
        (8.2, 45.9),
        (7.0, 45.9),
        (6.4, 46.15),
        (6.4, 47.0),
    ]
)
_overpass_coverage = {
    "DE": None,
    "LZ": None,
    "KU": None,
    "FR": _FRANCE_OVERPASS_COVERAGE,
    "CH": _SWITZERLAND_OVERPASS_COVERAGE,
}
_OSM_CACHE_MANIFEST_VERSION = 1

################################################################################
class OSM_layer:
    def __init__(self):
        self.dicosmn = (
            {}
        )  # keys are ints (ids) and values are tuple of (lat,lon)
        self.dicosmn_reverse = {}  # reverese of the previous one
        self.dicosmw = {}
        self.next_node_id = -1
        self.next_way_id = -1
        self.next_rel_id = -1
        # rels already sorted out and containing nodeids rather than wayids
        self.dicosmr = {}
        # original rels containing wayids only, not sorted and/or reversed
        self.dicosmrorig = {}
        # ids of objects directly queried, not of child or
        # parent objects pulled indirectly by queries. Since
        # osm ids are only unique per object type we need one for each:
        self.dicosmfirst = {"n": set(), "w": set(), "r": set()}
        self.dicosmtags = {"n": {}, "w": {}, "r": {}}
        self.dicosm = [
            self.dicosmn,
            self.dicosmw,
            self.dicosmr,
            self.dicosmrorig,
            self.dicosmfirst,
            self.dicosmtags,
        ]

    def reset(self):
        """Discard partially parsed data after a failed OSM input."""
        self.__init__()

    def update_dicosm(self, osm_input, input_tags=None, target_tags=None):
        try:
            result = self._update_dicosm(osm_input, input_tags, target_tags)
        except Exception as error:
            UI.vprint(0, "    OSM data could not be parsed:", error)
            result = 0
        if not result:
            self.reset()
        return int(bool(result))

    def _update_dicosm(self, osm_input, input_tags=None, target_tags=None):
        # input_tags (dict or None) are the input query tags (per osm type)
        # target_tags (dict or None) are the the tags which should be kept 
        # (per osm type) It is expected that if not None the target_tags 
        # contains the input_tags
        initnodes = len(self.dicosmn)
        initways = len(self.dicosmfirst["w"])
        initrels = len(self.dicosmfirst["r"])
        dicosmn_id_map = {}
        dicosmw_id_map = {}
        # osm_input may either refer to an osm filename (e.g. cached data) or
        # to a xml bytestring (direct download).  Validate the complete XML
        # before mutating the layer so a truncated response can never leave a
        # partially successful data set behind.
        try:
            if isinstance(osm_input, str):
                osm_file_name = osm_input
                if osm_file_name[-4:] == ".bz2":
                    with bz2.open(osm_file_name, "rb") as source:
                        payload = source.read()
                else:
                    with open(osm_file_name, "rb") as source:
                        payload = source.read()
            elif isinstance(osm_input, bytes):
                payload = osm_input
            else:
                UI.lvprint(0, "ERROR: OSM input must be a filename or bytes")
                return 0
            root = ElementTree.fromstring(payload)
            if root.tag.rsplit("}", 1)[-1].lower() != "osm":
                UI.lvprint(0, "ERROR: OSM input root is not <osm>")
                return 0
            # Overpass normally emits one element per line, but valid XML may
            # also be compacted into one line.  Re-serializing the validated
            # tree gives the existing line-oriented importer a stable shape
            # without accepting truncated input.
            if hasattr(ElementTree, "indent"):
                ElementTree.indent(root, space="  ")
            pfile = io.StringIO(ElementTree.tostring(root, encoding="unicode"))
        except Exception as error:
            UI.vprint(1, "    OSM input is corrupted or unreadable:", error)
            return 0
        first_line = pfile.readline()
        while first_line and "<osm" not in first_line.lower():
            first_line = pfile.readline()
        if "<osm" not in first_line.lower():
            pfile.close()
            UI.lvprint(0, "ERROR: OSM input has no opening <osm> tag")
            return 0
        def parse_element_line(line, element_name):
            """Parse one OSM element line without assuming quote style.

            Overpass normally emits one element per line, while a way or
            relation starts with a non-self-closing line.  ElementTree both
            accepts either XML quote style and performs exactly one entity
            unescape, so values such as ``&amp;quot;`` are not decoded twice.
            """
            text = line.strip()
            if not (
                text.startswith("<" + element_name + " ")
                or text.startswith("<" + element_name + ">")
            ):
                return None
            if not text.endswith("/>"):
                closing = text.rfind(">")
                if closing < 0:
                    return None
                text = text[: closing + 1] + "</" + element_name + ">"
            try:
                return ElementTree.fromstring(text)
            except ElementTree.ParseError:
                return None

        # ElementTree has already validated the complete document.  An empty
        # OSM document is commonly emitted as a single compact line, so its
        # closing tag was consumed together with the opening tag above and
        # will not appear in the line-oriented parser below.
        normal_exit = len(root) == 0
        for line in pfile:
            stripped_line = line.lstrip()
            if stripped_line.startswith("<node "):
                element = parse_element_line(line, "node")
                if element is None:
                    return 0
                osmtype = "n"
                osmid = element.attrib["id"]
                latp = float(element.attrib["lat"])
                lonp = float(element.attrib["lon"])
                if (lonp, latp) in self.dicosmn_reverse:
                    true_osmid = self.dicosmn_reverse[(lonp, latp)]
                    dicosmn_id_map[osmid] = true_osmid
                    osmid = true_osmid
                else:
                    true_osmid = self.next_node_id
                    dicosmn_id_map[osmid] = true_osmid
                    osmid = true_osmid
                    self.dicosmn_reverse[(lonp, latp)] = osmid
                    self.dicosmn[osmid] = (lonp, latp)
                    self.next_node_id -= 1
            elif stripped_line.startswith("<way "):
                element = parse_element_line(line, "way")
                if element is None:
                    return 0
                osmtype = "w"
                osmid = element.attrib["id"]
                true_osmid = self.next_way_id
                self.next_way_id -= 1
                dicosmw_id_map[osmid] = true_osmid
                osmid = true_osmid
                self.dicosmw[osmid] = []
                if not input_tags:
                    self.dicosmfirst["w"].add(osmid)
            elif stripped_line.startswith("<nd "):
                element = parse_element_line(line, "nd")
                if element is None:
                    return 0
                self.dicosmw[osmid].append(dicosmn_id_map[element.attrib["ref"]])
            elif stripped_line.startswith("<relation "):
                element = parse_element_line(line, "relation")
                if element is None:
                    return 0
                osmtype = "r"
                osmid = element.attrib["id"]
                true_osmid = self.next_rel_id
                self.next_rel_id -= 1
                osmid = true_osmid
                self.dicosmr[osmid] = {"outer": [], "inner": []}
                self.dicosmrorig[osmid] = {"outer": [], "inner": []}
                dico_rel_check = {"inner": {}, "outer": {}}
                if not input_tags:
                    self.dicosmfirst["r"].add(osmid)
            elif stripped_line.startswith("<member "):
                element = parse_element_line(line, "member")
                if element is None:
                    return 0
                member_type = element.attrib.get("type")
                role = element.attrib.get("role")
                if member_type != "way" or role not in ("outer", "inner"):
                    if member_type == "node":
                        continue  # not necessary to report these
                    UI.lvprint(
                        2,
                        "Relation id=",
                        osmid,
                        "contains a member of type",
                        "'" + str(member_type) + "'",
                        "and role",
                        "'" + role + "'",
                        "which was not treated (only deal with 'ways' of role ",
                        "'inner' or 'outer').",
                    )
                    continue
                try:
                    wayid = dicosmw_id_map[element.attrib["ref"]]
                except:
                    continue
                self.dicosmrorig[osmid][role].append(wayid)
                endpt1 = self.dicosmw[wayid][0]
                endpt2 = self.dicosmw[wayid][-1]
                if endpt1 == endpt2:
                    self.dicosmr[osmid][role].append(self.dicosmw[wayid])
                else:
                    if endpt1 in dico_rel_check[role]:
                        dico_rel_check[role][endpt1].append(wayid)
                    else:
                        dico_rel_check[role][endpt1] = [wayid]
                    if endpt2 in dico_rel_check[role]:
                        dico_rel_check[role][endpt2].append(wayid)
                    else:
                        dico_rel_check[role][endpt2] = [wayid]
            elif stripped_line.startswith("<tag "):
                element = parse_element_line(line, "tag")
                if element is None:
                    return 0
                # Do we need to catch that tag ?
                if (
                    (not input_tags)
                    or (("all", "") in target_tags[osmtype])
                    or ((element.attrib["k"], "") in target_tags[osmtype])
                    or ((element.attrib["k"], element.attrib["v"]) in target_tags[osmtype])
                ):
                    tag_key = element.attrib["k"]
                    tag_value = element.attrib["v"]
                    if osmid not in self.dicosmtags[osmtype]:
                        self.dicosmtags[osmtype][osmid] = {tag_key: tag_value}
                    else:
                        self.dicosmtags[osmtype][osmid][tag_key] = tag_value
                    # If so, do we need to declare this osmid as a first catch, 
                    # not one only brought with as a child
                    if input_tags and (
                        ((tag_key, "") in input_tags[osmtype])
                        or ((tag_key, tag_value) in input_tags[osmtype])
                    ):
                        self.dicosmfirst[osmtype].add(osmid)
            elif "</way" in stripped_line:
                if not self.dicosmw[osmid]:
                    del self.dicosmw[osmid]
                    self.next_way_id += 1
                    if osmid in self.dicosmfirst["w"]:
                        self.dicosmfirst["w"].remove(osmid)
                    if osmid in self.dicosmtags["w"]:
                        del self.dicosmtags[osmtype][osmid]
            elif "</relation>" in stripped_line:
                bad_rel = False
                for role, endpt in (
                    (r, e)
                    for r in ["outer", "inner"]
                    for e in dico_rel_check[r]
                ):
                    if len(dico_rel_check[role][endpt]) != 2:
                        bad_rel = True
                        break
                if bad_rel == True:
                    UI.lvprint(
                        2,
                        "Relation id=",
                        osmid,
                        "is ill formed and was not treated.",
                    )
                    del self.dicosmr[osmid]
                    del self.dicosmrorig[osmid]
                    del dico_rel_check
                    self.next_rel_id += 1
                    if osmid in self.dicosmfirst["r"]:
                        self.dicosmfirst["r"].remove(osmid)
                    if osmid in self.dicosmtags["r"]:
                        del self.dicosmtags["r"][osmid]
                    continue
                for role in ["outer", "inner"]:
                    while dico_rel_check[role]:
                        nodeids = []
                        endpt = next(iter(dico_rel_check[role]))
                        wayid = dico_rel_check[role][endpt][0]
                        endptinit = self.dicosmw[wayid][0]
                        endpt1 = endptinit
                        endpt2 = self.dicosmw[wayid][-1]
                        for nodeid in self.dicosmw[wayid][:-1]:
                            nodeids.append(nodeid)
                        while endpt2 != endptinit:
                            if dico_rel_check[role][endpt2][0] == wayid:
                                wayid = dico_rel_check[role][endpt2][1]
                            else:
                                wayid = dico_rel_check[role][endpt2][0]
                            endpt1 = endpt2
                            if self.dicosmw[wayid][0] == endpt1:
                                endpt2 = self.dicosmw[wayid][-1]
                                for nodeid in self.dicosmw[wayid][:-1]:
                                    nodeids.append(nodeid)
                            else:
                                endpt2 = self.dicosmw[wayid][0]
                                for nodeid in self.dicosmw[wayid][-1:0:-1]:
                                    nodeids.append(nodeid)
                            del dico_rel_check[role][endpt1]
                        nodeids.append(endptinit)
                        self.dicosmr[osmid][role].append(nodeids)
                        del dico_rel_check[role][endptinit]
                if target_tags == None:
                    for wayid in (
                        self.dicosmrorig[osmid]["outer"]
                        + self.dicosmrorig[osmid]["inner"]
                    ):
                        try:
                            self.dicosmfirst["w"].remove(wayid)
                        except:
                            pass
                if not self.dicosmr[osmid]["outer"]:
                    del self.dicosmr[osmid]
                    del self.dicosmrorig[osmid]
                    self.next_rel_id += 1
                    if osmid in self.dicosmfirst["r"]:
                        self.dicosmfirst["r"].remove(osmid)
                    if osmid in self.dicosmtags["r"]:
                        del self.dicosmtags["r"][osmid]
                del dico_rel_check
            elif "</osm>" in stripped_line.lower():
                normal_exit = True
        pfile.close()
        if not normal_exit:
            UI.lvprint(
                0,
                "ERROR: OSM overpass server answer was corrupted ",
                "(no ending </OSM> tag)",
            )
            return 0
        UI.vprint(
            2,
            "      A total of "
            + str(len(self.dicosmn) - initnodes)
            + " new node(s), "
            + str(len(self.dicosmfirst["w"]) - initways)
            + " new ways and "
            + str(len(self.dicosmfirst["r"]) - initrels)
            + " new relation(s).",
        )
        return 1

    def write_to_file(self, filename):
        temporary_filename = (
            filename + ".tmp.bz2"
            if filename.endswith(".bz2")
            else filename + ".tmp"
        )
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            result = self._write_to_file(temporary_filename)
            if not result:
                return 0
            os.replace(temporary_filename, filename)
            return 1
        except Exception as error:
            UI.vprint(1, "    Could not atomically write", filename, ":", error)
            try:
                os.remove(temporary_filename)
            except OSError:
                pass
            return 0

    def _write_to_file(self, filename):
        try:
            if filename[-4:] == ".bz2":
                fout = bz2.open(filename, "wt", encoding="utf-8")
            else:
                fout = open(filename, "w", encoding="utf-8")
        except:
            UI.vprint(1, "    Could not open", filename, "for writing.")
            return 0
        fout.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6" ' + 
            'generator="Ortho4XP">\n'
        )
        for nodeid in sorted(self.dicosmn, key=stable_id_key):
            (lonp, latp) = self.dicosmn[nodeid]
            node_tags = self.dicosmtags["n"].get(nodeid)
            if not node_tags:
                fout.write(
                    '  <node id="'
                    + str(nodeid)
                    + '" lat="'
                    + "{:.7f}".format(latp)
                    + '" lon="'
                    + "{:.7f}".format(lonp)
                    + '" version="1"/>\n'
                )
                continue
            fout.write(
                '  <node id="'
                + str(nodeid)
                + '" lat="'
                + "{:.7f}".format(latp)
                + '" lon="'
                + "{:.7f}".format(lonp)
                + '" version="1">\n'
            )
            for tag, value in sorted(node_tags.items(), key=lambda item: str(item[0])):
                fout.write(
                    "    <tag k="
                    + quoteattr(str(tag))
                    + " v="
                    + quoteattr(str(value))
                    + "/>\n"
                )
            fout.write("  </node>\n")
        first_way_ids = sorted(self.dicosmfirst["w"], key=stable_id_key)
        remaining_way_ids = sorted(
            set(self.dicosmw).difference(self.dicosmfirst["w"]),
            key=stable_id_key,
        )
        for wayid in tuple(first_way_ids) + tuple(remaining_way_ids):
            fout.write('  <way id="' + str(wayid) + '" version="1">\n')
            for nodeid in self.dicosmw[wayid]:
                fout.write('    <nd ref="' + str(nodeid) + '"/>\n')
            for tag in sorted(
                self.dicosmtags["w"][wayid]
                if wayid in self.dicosmtags["w"]
                else [],
                key=str,
            ):
                fout.write(
                    "    <tag k="
                    + quoteattr(str(tag))
                    + " v="
                    + quoteattr(str(self.dicosmtags["w"][wayid][tag]))
                    + "/>\n"
                )
            fout.write("  </way>\n")
        first_relation_ids = sorted(self.dicosmfirst["r"], key=stable_id_key)
        remaining_relation_ids = sorted(
            set(self.dicosmrorig).difference(self.dicosmfirst["r"]),
            key=stable_id_key,
        )
        for relid in tuple(first_relation_ids) + tuple(remaining_relation_ids):
            fout.write('  <relation id="' + str(relid) + '" version="1">\n')
            for wayid in self.dicosmrorig[relid]["outer"]:
                fout.write(
                    '    <member type="way" ref="'
                    + str(wayid)
                    + '" role="outer"/>\n'
                )
            for wayid in self.dicosmrorig[relid]["inner"]:
                fout.write(
                    '    <member type="way" ref="'
                    + str(wayid)
                    + '" role="inner"/>\n'
                )
            for tag in sorted(
                self.dicosmtags["r"][relid]
                if relid in self.dicosmtags["r"]
                else [],
                key=str,
            ):
                fout.write(
                    "    <tag k="
                    + quoteattr(str(tag))
                    + " v="
                    + quoteattr(str(self.dicosmtags["r"][relid][tag]))
                    + "/>\n"
                )
            fout.write("  </relation>\n")
        fout.write("</osm>")
        fout.close()
        return 1

################################################################################
def _quarantine_osm_cache(filename):
    """Move an invalid cache aside without destroying the original data."""
    if not filename or not os.path.exists(filename):
        return True
    candidate = filename + ".bad"
    suffix = 1
    while os.path.exists(candidate):
        candidate = filename + ".bad." + str(suffix)
        suffix += 1
    try:
        os.replace(filename, candidate)
    except OSError as error:
        UI.vprint(0, "    Could not quarantine corrupted OSM cache:", error)
        return False
    UI.vprint(1, "    Quarantined corrupted OSM cache as", candidate)
    return True


def _preserve_unverified_osm_cache(filename, manifest_filename):
    """Move an unverified cache generation aside before publishing a new one."""
    moved = []
    try:
        candidate = filename + ".unverified"
        suffix = 1
        while os.path.exists(candidate) or os.path.exists(candidate + ".manifest.json"):
            candidate = filename + ".unverified." + str(suffix)
            suffix += 1
        if os.path.isfile(filename):
            os.replace(filename, candidate)
            moved.append((candidate, filename))
        if os.path.isfile(manifest_filename):
            manifest_candidate = candidate + ".manifest.json"
            os.replace(manifest_filename, manifest_candidate)
            moved.append((manifest_candidate, manifest_filename))
        if moved:
            UI.vprint(1, "    Preserved unverified OSM cache generation as", candidate)
        return True
    except OSError as error:
        UI.vprint(0, "    Could not preserve unverified OSM cache:", error)
        # Restore any moves made in this operation. Do not leave a half-paired
        # data/manifest generation behind.
        for source, destination in reversed(moved):
            try:
                os.replace(source, destination)
            except OSError:
                pass
        return False


def _update_cached_osm(osm_layer, filename, input_tags, target_tags):
    try:
        return bool(osm_layer.update_dicosm(filename, input_tags, target_tags))
    except Exception as error:
        UI.vprint(0, "    Cached OSM data could not be parsed:", error)
        return False


def _build_osm_tag_filters(queries, tags_of_interest=None):
    """Build the parser filters used by both tile and standalone OSM loads."""
    queries = [] if queries is None else list(queries)
    if tags_of_interest is None:
        tags_of_interest = []
    elif isinstance(tags_of_interest, str):
        tags_of_interest = [tags_of_interest]
    else:
        tags_of_interest = list(tags_of_interest)

    target_tags = {"n": [], "w": [], "r": []}
    input_tags = {"n": [], "w": [], "r": []}
    for query in queries:
        query_parts = [query] if isinstance(query, str) else query
        for tag in query_parts:
            if not isinstance(tag, str):
                continue
            items = tag.split('"')
            if not items or not items[0]:
                continue
            osm_type = items[0][0]
            if osm_type not in target_tags:
                continue
            try:
                query_tag = (items[1], items[3])
            except IndexError:
                query_tag = (items[1], "") if len(items) > 1 else None
            if query_tag is None:
                continue
            input_tags[osm_type].append(query_tag)
            if query_tag not in target_tags[osm_type]:
                target_tags[osm_type].append(query_tag)
            for interest in tags_of_interest:
                if isinstance(interest, str):
                    interest_tag = (interest, "")
                elif isinstance(interest, (tuple, list)) and len(interest) == 2:
                    interest_tag = tuple(interest)
                else:
                    continue
                if interest_tag not in target_tags[osm_type]:
                    target_tags[osm_type].append(interest_tag)
    return input_tags, target_tags


def normalize_osm_failure_policy(value=None):
    policy = osm_download_failure_policy if value is None else value
    return policy if policy in OSM_FAILURE_POLICIES else "abort"


def _normalized_bbox(bbox):
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return None
    try:
        return [round(float(value), 7) for value in bbox]
    except (TypeError, ValueError):
        return None


def _bbox_is_covered(server_code, bbox):
    coverage = _overpass_coverage.get(server_code)
    normalized = _normalized_bbox(bbox)
    if coverage is None:
        return True
    if normalized is None:
        return False
    south, west, north, east = normalized
    if south >= north or west >= east:
        return False
    return coverage.covers(geometry.box(west, south, east, north))


def _server_order(preferred_server, bbox):
    server_order = ["DE", "LZ", "CH", "FR", "KU"]
    if preferred_server == "random":
        random.shuffle(server_order)
    elif preferred_server in overpass_servers:
        server_order.remove(preferred_server)
        server_order.insert(0, preferred_server)
    else:
        UI.vprint(0, "ERROR: Unknown Overpass server:", preferred_server)
        return []

    eligible = []
    for server_code in server_order:
        if _bbox_is_covered(server_code, bbox):
            eligible.append(server_code)
            continue
        UI.logprint(
            "[OSM] skipped_server=",
            server_code,
            "reason=bbox-not-covered",
            "bbox=",
            _normalized_bbox(bbox),
        )
        UI.vprint(
            2,
            "        Skipping Overpass server",
            server_code,
            "because it does not cover bbox",
            _normalized_bbox(bbox),
        )
    return eligible


def _query_signature(queries, tags_of_interest=None):
    canonical_queries = []
    for query in queries:
        if isinstance(query, (tuple, list)):
            canonical_queries.append([str(item) for item in query])
        else:
            canonical_queries.append(str(query))
    if tags_of_interest is None:
        tags_of_interest = []
    elif isinstance(tags_of_interest, str):
        tags_of_interest = [tags_of_interest]
    payload = {
        "queries": canonical_queries,
        "tags_of_interest": [
            list(item) if isinstance(item, tuple) else item
            for item in (tags_of_interest or [])
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(filename):
    digest = hashlib.sha256()
    with open(filename, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _layer_counts(osm_layer):
    return {
        "nodes": len(osm_layer.dicosmn),
        "ways": len(osm_layer.dicosmfirst["w"]),
        "relations": len(osm_layer.dicosmfirst["r"]),
        "elements": (
            len(osm_layer.dicosmn)
            + len(osm_layer.dicosmfirst["w"])
            + len(osm_layer.dicosmfirst["r"])
        ),
    }


def _replace_layer(destination, source):
    destination.__dict__.clear()
    destination.__dict__.update(source.__dict__)


def _record_layer_failure(osm_layer, failure):
    osm_layer.reset()
    osm_layer.last_result = OSM_FAILED
    osm_layer.last_failure = failure
    osm_layer.last_cache_info = None


def _write_json_atomic(filename, payload):
    temporary_filename = filename + ".tmp"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        with open(temporary_filename, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_filename, filename)
        return True
    except OSError as error:
        UI.vprint(1, "    Could not atomically write OSM metadata:", filename, error)
        try:
            os.remove(temporary_filename)
        except OSError:
            pass
        return False


def _build_cache_manifest(
    layer_name, bbox, queries, tags_of_interest, responses, osm_layer, filename
):
    response_statuses = [response.get("data_status") for response in responses]
    validity = VALID_EMPTY if response_statuses and all(
        status == VALID_EMPTY for status in response_statuses
    ) else VALID_DATA
    return {
        "schema_version": _OSM_CACHE_MANIFEST_VERSION,
        "status": "complete",
        "validity": validity,
        "layer": layer_name,
        "bbox": _normalized_bbox(bbox),
        "query_signature": _query_signature(queries, tags_of_interest),
        "queries": [
            [str(item) for item in query] if isinstance(query, (tuple, list)) else str(query)
            for query in queries
        ],
        "responses": responses,
        "counts": _layer_counts(osm_layer),
        "data_sha256": _sha256_file(filename),
    }


def _load_verified_cache(
    osm_layer, filename, manifest_filename, layer_name, bbox, queries, tags_of_interest,
    input_tags, target_tags, allow_unbound_request=False,
):
    if not os.path.isfile(filename) or not os.path.isfile(manifest_filename):
        return False
    try:
        with open(manifest_filename, "r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict):
            UI.vprint(2, "    OSM cache manifest is not an object:", manifest_filename)
            return False
        if manifest.get("schema_version") != _OSM_CACHE_MANIFEST_VERSION:
            return False
        if manifest.get("status") != "complete":
            return False
        if manifest.get("layer") != layer_name:
            return False
        if allow_unbound_request:
            if manifest.get("bbox") is not None:
                return False
        else:
            if manifest.get("bbox") != _normalized_bbox(bbox):
                return False
            if manifest.get("query_signature") != _query_signature(
                queries, tags_of_interest
            ):
                return False
        responses = manifest.get("responses")
        if not isinstance(responses, list) or not responses:
            return False
        if not allow_unbound_request and len(responses) != len(queries):
            return False
        response_statuses = []
        for response in responses:
            if not isinstance(response, dict):
                return False
            data_status = response.get("data_status")
            if data_status not in (VALID_DATA, VALID_EMPTY):
                return False
            if response.get("status") != data_status:
                return False
            if response.get("http_status") != 200:
                return False
            response_server = response.get("server")
            response_bbox = manifest.get("bbox") if allow_unbound_request else bbox
            if response_server not in overpass_servers or not _bbox_is_covered(
                response_server, response_bbox
            ):
                return False
            response_statuses.append(data_status)
        validity = manifest.get("validity")
        if validity not in (VALID_DATA, VALID_EMPTY):
            return False
        if validity == VALID_EMPTY and any(
            status != VALID_EMPTY for status in response_statuses
        ):
            return False
        if manifest.get("data_sha256") != _sha256_file(filename):
            return False

        candidate = OSM_layer()
        if not _update_cached_osm(candidate, filename, input_tags, target_tags):
            return False
        if manifest.get("counts") != _layer_counts(candidate):
            return False
        _replace_layer(osm_layer, candidate)
        osm_layer.last_result = OSM_COMPLETE
        osm_layer.last_failure = None
        osm_layer.last_cache_info = {
            "source": "verified-cache",
            "manifest": manifest_filename,
            "validity": manifest.get("validity", VALID_DATA),
            "counts": manifest.get("counts", {}),
        }
        UI.vprint(1, "    * Recycling verified OSM data from", filename)
        UI.logprint(
            "[OSM] cache=verified layer=",
            layer_name,
            "manifest=",
            manifest_filename,
        )
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        UI.vprint(2, "    OSM cache metadata is invalid:", manifest_filename, error)
        return False


def load_verified_osm_cache(
    queries, osm_layer, lat, lon, tags_of_interest=None, cached_suffix=""
):
    """Load the exact verified cache for one tile/layer, if available."""
    queries = list(queries)
    if tags_of_interest is None:
        tags_of_interest = []
    elif isinstance(tags_of_interest, str):
        tags_of_interest = [tags_of_interest]
    input_tags, target_tags = _build_osm_tag_filters(queries, tags_of_interest)
    bbox = (lat, lon, lat + 1, lon + 1)
    if not cached_suffix:
        return False
    filename = FNAMES.osm_cached(lat, lon, cached_suffix)
    manifest_filename = FNAMES.osm_cache_manifest(lat, lon, cached_suffix)
    return _load_verified_cache(
        osm_layer,
        filename,
        manifest_filename,
        cached_suffix,
        bbox,
        queries,
        tags_of_interest,
        input_tags,
        target_tags,
    )


def _retry_after_seconds(response):
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, min(float(value), 300.0))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
            if retry_at.tzinfo is None:
                return None
            from datetime import datetime, timezone
            return max(0.0, min((retry_at - datetime.now(timezone.utc)).total_seconds(), 300.0))
        except (TypeError, ValueError, OverflowError):
            return None


def _retry_delay(round_index, retry_after_values, retry_statuses):
    if retry_after_values:
        return max(retry_after_values)
    if any(status in (406, 429) for status in retry_statuses):
        return 30.0
    base = (5.0, 15.0, 30.0)[min(round_index, 2)]
    return base * random.uniform(0.8, 1.2)


def prompt_osm_failure(tile, failure, cache_available=False):
    """Ask once per tile without touching Tk from the worker thread."""
    gui = getattr(UI, "gui", None)
    if gui is None or not hasattr(gui, "after"):
        return "abort"

    choice = {"value": "abort"}
    completed = threading.Event()

    def show_dialog():
        dialog = None
        try:
            dialog = tk.Toplevel(gui)
            dialog.title(UI.ui_text("OSM download failed", "OSMデータの取得に失敗しました"))
            dialog.transient(gui)
            dialog.grab_set()
            detail = "{}: {}".format(
                failure.get("layer", "OSM"),
                failure.get("query", "unknown query"),
            )
            tk.Label(
                dialog,
                text=UI.ui_text(
                    "No valid OSM response was received after retries.\n" + detail,
                    "再試行後も有効なOSM応答を受信できませんでした。\n" + detail,
                ),
                justify="left",
                padx=16,
                pady=12,
            ).pack(fill="x")

            buttons = [("Retry", "retry")]
            if cache_available:
                buttons.append(("Use verified cache", "use_cache"))
            buttons.extend(
                [
                    ("Continue as degraded", "continue_degraded"),
                    ("Stop tile", "abort"),
                ]
            )

            def select(value):
                choice["value"] = value
                try:
                    dialog.grab_release()
                    dialog.destroy()
                except tk.TclError:
                    pass
                completed.set()

            for english, value in buttons:
                tk.Button(
                    dialog,
                    text=UI.ui_text(
                        english,
                        {
                            "Retry": "再試行",
                            "Use verified cache": "検証済みキャッシュを使用",
                            "Continue as degraded": "欠落扱いで継続",
                            "Stop tile": "タイルを停止",
                        }[english],
                    ),
                    command=lambda value=value: select(value),
                ).pack(fill="x", padx=16, pady=3)
            dialog.protocol("WM_DELETE_WINDOW", lambda: select("abort"))
        except Exception as error:
            UI.logprint("[OSM] failure dialog unavailable:", repr(error))
            choice["value"] = "abort"
            completed.set()

    try:
        gui.after(0, show_dialog)
    except Exception:
        return "abort"
    completed.wait()
    return choice["value"]


def run_osm_layer_with_policy(tile, layer_name, queries, osm_layer, **kwargs):
    """Run a tile OSM layer and apply the configured failure policy."""
    queries = list(queries)
    cached_suffix = kwargs.get("cached_suffix", "")
    tags_of_interest = kwargs.get("tags_of_interest", None)

    while True:
        result = OSM_queries_to_OSM_layer(
            queries,
            osm_layer,
            tile.lat,
            tile.lon,
            **kwargs,
        )
        if result == OSM_COMPLETE:
            return OSM_COMPLETE
        if result == OSM_DEGRADED:
            return OSM_DEGRADED

        failure = dict(getattr(osm_layer, "last_failure", {}) or {})
        failure["layer"] = layer_name
        cache_info = getattr(osm_layer, "last_cache_info", None)
        cache = {
            "used": bool(cache_info),
            "source": (cache_info or {}).get("source", "none"),
            "data": FNAMES.osm_cached(tile.lat, tile.lon, cached_suffix)
            if cached_suffix
            else None,
            "manifest": FNAMES.osm_cache_manifest(
                tile.lat, tile.lon, cached_suffix
            )
            if cached_suffix
            else None,
            "available": False,
        }
        failure["cache"] = cache
        failures = getattr(tile, "osm_failures", None)
        if failures is None:
            failures = []
            tile.osm_failures = failures
        failures.append(failure)

        action = getattr(tile, "osm_failure_action", None)
        if action is None:
            policy = normalize_osm_failure_policy()
            if policy == "prompt":
                cache_probe = OSM_layer()
                cache_available = bool(
                    cached_suffix
                    and load_verified_osm_cache(
                        queries,
                        cache_probe,
                        tile.lat,
                        tile.lon,
                        tags_of_interest=tags_of_interest,
                        cached_suffix=cached_suffix,
                    )
                )
                cache["available"] = cache_available
                action = prompt_osm_failure(
                    tile, failure, cache_available=cache_available
                )
            else:
                action = policy
            if action != "retry":
                tile.osm_failure_action = action

        if action == "retry":
            osm_layer.reset()
            continue
        if action == "use_cache":
            cached_layer = OSM_layer()
            if cached_suffix and load_verified_osm_cache(
                queries,
                cached_layer,
                tile.lat,
                tile.lon,
                tags_of_interest=tags_of_interest,
                cached_suffix=cached_suffix,
            ):
                _replace_layer(osm_layer, cached_layer)
                cache["used"] = True
                cache["available"] = True
                cache["source"] = "verified-cache"
                return OSM_COMPLETE
            failure["reason"] = "verified-cache-unavailable"
            _record_layer_failure(osm_layer, failure)
            return OSM_FAILED
        if action == "continue_degraded":
            osm_layer.reset()
            if not hasattr(tile, "osm_degraded_layers"):
                tile.osm_degraded_layers = set()
            tile.osm_degraded_layers.add(layer_name)
            UI.vprint(
                0,
                UI.ui_text(
                    "WARNING: OSM layer {} is unavailable; continuing as degraded.".format(
                        layer_name
                    ),
                    "警告: OSMレイヤー{}を取得できないため、欠落扱いで継続します。".format(
                        layer_name
                    ),
                ),
            )
            return OSM_DEGRADED

        return OSM_FAILED


################################################################################
def OSM_queries_to_OSM_layer(
    queries,
    osm_layer,
    lat,
    lon,
    tags_of_interest=None,
    server_code=None,
    cached_suffix="",
):
    # Keep every query in a temporary layer. A cache and its manifest are
    # published only after the complete query set has succeeded.
    queries = list(queries)
    if tags_of_interest is None:
        tags_of_interest = []
    elif isinstance(tags_of_interest, str):
        tags_of_interest = [tags_of_interest]
    else:
        tags_of_interest = list(tags_of_interest)
    input_tags, target_tags = _build_osm_tag_filters(queries, tags_of_interest)
    bbox = (lat, lon, lat + 1, lon + 1)
    cached_data_filename = FNAMES.osm_cached(lat, lon, cached_suffix)
    manifest_filename = FNAMES.osm_cache_manifest(lat, lon, cached_suffix)
    osm_layer.last_result = OSM_FAILED
    osm_layer.last_failure = None
    osm_layer.last_cache_info = None

    if cached_suffix and _load_verified_cache(
        osm_layer,
        cached_data_filename,
        manifest_filename,
        cached_suffix,
        bbox,
        queries,
        tags_of_interest,
        input_tags,
        target_tags,
    ):
        return OSM_COMPLETE

    if cached_suffix and os.path.isfile(cached_data_filename):
        UI.vprint(
            1,
            "    * Ignoring unverified OSM cache (manifest missing or mismatched):",
            cached_data_filename,
        )
        UI.logprint(
            "[OSM] cache=unverified layer=",
            cached_suffix,
            "data=",
            cached_data_filename,
        )

    unverified_cache_present = bool(
        cached_suffix
        and (
            os.path.isfile(cached_data_filename)
            or os.path.isfile(manifest_filename)
        )
    )

    # Legacy per-query caches have no query/bbox/completeness proof. Keep
    # them on disk for manual recovery, but never mix them into a new layer.
    for query in queries:
        if not isinstance(query, str):
            continue
        old_cached_data_filename = FNAMES.osm_old_cached(lat, lon, query)
        if os.path.isfile(old_cached_data_filename):
            UI.vprint(
                2,
                "    * Ignoring legacy unverified OSM cache:",
                old_cached_data_filename,
            )

    candidate_layer = OSM_layer()
    responses = []
    for query in queries:
        UI.vprint(1, "    * Downloading OSM data for", query)
        response, response_info = get_overpass_data(
            query, bbox, server_code, return_metadata=True
        )
        if UI.red_flag:
            _record_layer_failure(osm_layer, {
                "layer": cached_suffix or "OSM",
                "query": _overpass_query_label(query),
                "reason": "cancelled",
                "metadata": response_info,
            })
            return 0
        if not response:
            UI.logprint(
                "No valid answer for",
                query,
                "after",
                max_osm_tentatives,
                ".",
            )
            UI.vprint(
                1,
                "      No valid answer after",
                max_osm_tentatives,
                "; layer will not be published.",
            )
            _record_layer_failure(osm_layer, {
                "layer": cached_suffix or "OSM",
                "query": _overpass_query_label(query),
                "reason": "no-valid-response",
                "metadata": response_info,
            })
            return 0
        if not candidate_layer.update_dicosm(response, input_tags, target_tags):
            _record_layer_failure(osm_layer, {
                "layer": cached_suffix or "OSM",
                "query": _overpass_query_label(query),
                "reason": "response-parse-failed",
                "metadata": response_info,
            })
            return 0
        responses.append(response_info)

    _replace_layer(osm_layer, candidate_layer)
    osm_layer.last_result = OSM_COMPLETE
    if cached_suffix:
        can_publish_cache = not unverified_cache_present or _preserve_unverified_osm_cache(
            cached_data_filename, manifest_filename
        )
        if not can_publish_cache:
            UI.vprint(
                1,
                "    WARNING: Keeping the new OSM data in memory; cache publication was skipped.",
            )
        elif not osm_layer.write_to_file(cached_data_filename):
            UI.vprint(1, "    WARNING: Could not save OSM cache", cached_data_filename)
        else:
            manifest = _build_cache_manifest(
                cached_suffix,
                bbox,
                queries,
                tags_of_interest,
                responses,
                osm_layer,
                cached_data_filename,
            )
            if not _write_json_atomic(manifest_filename, manifest):
                UI.vprint(1, "    WARNING: OSM cache remains unverified:", cached_data_filename)
    osm_layer.last_cache_info = {
        "source": "network",
        "responses": responses,
        "counts": _layer_counts(osm_layer),
    }
    return OSM_COMPLETE

################################################################################
def OSM_query_to_OSM_layer(
    query,
    bbox,
    osm_layer,
    tags_of_interest=None,
    server_code=None,
    cached_file_name="",
):
    # This helper is used by the standalone mask/extent command.  It must use
    # the same manifest contract as tile OSM caches, while allowing an
    # explicitly requested cache-only invocation to omit the original query.
    if tags_of_interest is None:
        tags_of_interest = []
    elif isinstance(tags_of_interest, str):
        tags_of_interest = [tags_of_interest]
    else:
        tags_of_interest = list(tags_of_interest)
    query_list = (
        []
        if query is None
        else [query]
        if isinstance(query, str)
        else list(query)
    )
    if query is None:
        input_tags = None
        target_tags = None
    else:
        input_tags, target_tags = _build_osm_tag_filters(
            query_list, tags_of_interest
        )

    manifest_filename = (
        cached_file_name + ".manifest.json" if cached_file_name else ""
    )
    if cached_file_name:
        verified_layer = OSM_layer()
        if _load_verified_cache(
            verified_layer,
            cached_file_name,
            manifest_filename,
            "standalone",
            bbox,
            query_list,
            tags_of_interest,
            input_tags,
            target_tags,
            allow_unbound_request=query is None,
        ):
            _replace_layer(osm_layer, verified_layer)
            return 1
        if os.path.isfile(cached_file_name) or os.path.isfile(manifest_filename):
            UI.vprint(
                1,
                "    * Ignoring unverified standalone OSM cache:",
                cached_file_name,
            )
        if query is None:
            _record_layer_failure(
                osm_layer,
                {
                    "layer": "standalone",
                    "reason": "verified-cache-required",
                },
            )
            return 0

    if query is None:
        _record_layer_failure(
            osm_layer,
            {"layer": "standalone", "reason": "query-required"},
        )
        return 0

    response_result = get_overpass_data(
        query, bbox, server_code, return_metadata=True
    )
    if isinstance(response_result, tuple) and len(response_result) == 2:
        response, response_info = response_result
    else:
        response = response_result
        response_info = {}
    if UI.red_flag:
        _record_layer_failure(
            osm_layer,
            {"layer": "standalone", "reason": "cancelled"},
        )
        return 0
    if not response:
        UI.lvprint(
            1,
            "      No valid answer for",
            query,
            "after",
            max_osm_tentatives,
            ", skipping it.",
        )
        _record_layer_failure(
            osm_layer,
            {
                "layer": "standalone",
                "query": _overpass_query_label(query),
                "reason": "no-valid-response",
                "metadata": response_info,
            },
        )
        return 0

    data_status, reason, counts = _inspect_osm_response(response)
    if data_status is None:
        _record_layer_failure(
            osm_layer,
            {
                "layer": "standalone",
                "query": _overpass_query_label(query),
                "reason": reason or "response-parse-failed",
                "metadata": response_info,
            },
        )
        return 0
    response_info = dict(response_info or {})
    response_info.setdefault("status", data_status)
    response_info.setdefault("data_status", data_status)
    response_info.setdefault("http_status", 200)
    response_info.setdefault("counts", counts)
    response_info.setdefault("server", server_code or overpass_server_choice)

    candidate_layer = OSM_layer()
    if not candidate_layer.update_dicosm(response, input_tags, target_tags):
        _record_layer_failure(
            osm_layer,
            {
                "layer": "standalone",
                "query": _overpass_query_label(query),
                "reason": "response-parse-failed",
                "metadata": response_info,
            },
        )
        return 0
    _replace_layer(osm_layer, candidate_layer)
    osm_layer.last_result = OSM_COMPLETE
    osm_layer.last_failure = None

    if cached_file_name:
        has_unverified_cache = os.path.isfile(cached_file_name) or os.path.isfile(
            manifest_filename
        )
        can_publish_cache = not has_unverified_cache or _preserve_unverified_osm_cache(
            cached_file_name, manifest_filename
        )
        if can_publish_cache and osm_layer.write_to_file(cached_file_name):
            manifest = _build_cache_manifest(
                "standalone",
                bbox,
                query_list,
                tags_of_interest,
                [response_info],
                osm_layer,
                cached_file_name,
            )
            if not _write_json_atomic(manifest_filename, manifest):
                UI.vprint(
                    1,
                    "    WARNING: OSM cache remains unverified:",
                    cached_file_name,
                )
        elif not can_publish_cache:
            UI.vprint(
                1,
                "    WARNING: Keeping standalone OSM data in memory; cache publication was skipped.",
            )
        else:
            UI.vprint(1, "    WARNING: Could not save OSM cache", cached_file_name)
    osm_layer.last_cache_info = {
        "source": "network",
        "responses": [response_info],
        "counts": _layer_counts(osm_layer),
    }
    return 1

################################################################################
def _overpass_query_label(query):
    """Return a compact label suitable for retry diagnostics."""
    if isinstance(query, (list, tuple)):
        label = " | ".join(str(item) for item in query)
    else:
        label = str(query)
    label = " ".join(label.split())
    return label if len(label) <= 160 else label[:157] + "..."


def _inspect_osm_response(content):
    if not content:
        return None, "missing closing </osm> tag", None
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError:
        reason = (
            "missing closing </osm> tag"
            if b"</osm>" not in content.lower()
            else "malformed XML response"
        )
        return None, reason, None
    if root.tag.rsplit("}", 1)[-1].lower() != "osm":
        return None, "unexpected XML root", None
    if any(child.tag.rsplit("}", 1)[-1].lower() == "remark" for child in root):
        return None, "server remark indicates incomplete/error response", None
    counts = {"node": 0, "way": 0, "relation": 0}
    for child in root:
        tag = child.tag.rsplit("}", 1)[-1].lower()
        if tag in counts:
            counts[tag] += 1
    total = sum(counts.values())
    return (VALID_EMPTY if total == 0 else VALID_DATA), None, counts


def get_overpass_data(query, bbox, server_code=None, return_metadata=False):
    preferred_server = server_code or overpass_server_choice
    server_order = _server_order(preferred_server, bbox)
    if not server_order:
        metadata = {"status": FAILED, "reason": "no-covered-server"}
        return (None, metadata) if return_metadata else 0

    # rel を relation に置換（互換性のため）
    clean_query = (
        query.replace("rel[", "relation[")
        if isinstance(query, str)
        else [q.replace("rel[", "relation[") for q in query]
    )
    if isinstance(query, str):
        overpass_query = clean_query + str(bbox) + ";"
    else:  # query is a tuple
        overpass_query = "".join([x + str(bbox) + ";" for x in clean_query])
    full_query = "[timeout:300];(" + overpass_query + ");(._;>>;);out meta;"
    headers = {"User-Agent": "Ortho4XP"}
    session = requests.Session()
    query_label = _overpass_query_label(query)
    attempts = []

    for tentative in range(max(1, int(max_osm_tentatives))):
        retry_after_values = []
        retry_statuses = []
        for true_server_code in server_order:
            base_url = overpass_servers[true_server_code]
            attempt_number = tentative + 1
            UI.logprint(
                "[OSM] query=",
                query_label,
                "attempt=",
                attempt_number,
                "server=",
                true_server_code,
            )
            UI.vprint(3, "Sending POST request to", base_url)
            try:
                # POST keeps large vector queries out of URL length limits.
                response = session.post(
                    base_url,
                    data={"data": full_query},
                    timeout=310,
                    headers=headers,
                )
                status_code = getattr(response, "status_code", None)
                UI.vprint(3, "OSM response status :", status_code)
                content = response.content or b""
                if status_code == 200:
                    data_status, reason, counts = _inspect_osm_response(content)
                    if data_status is not None:
                        metadata = {
                            "status": data_status,
                            "data_status": data_status,
                            "server": true_server_code,
                            "attempt": attempt_number,
                            "http_status": 200,
                            "counts": counts,
                            "payload_sha256": hashlib.sha256(content).hexdigest(),
                            "attempts": attempts,
                        }
                        UI.logprint(
                            "[OSM] query=",
                            query_label,
                            "success_server=",
                            true_server_code,
                            "attempt=",
                            attempt_number,
                            "status=200",
                            "data_status=",
                            data_status,
                        )
                        UI.vprint(
                            2,
                            "        OSM query succeeded on server",
                            true_server_code,
                            "(attempt",
                            attempt_number,
                            "):",
                            query_label,
                            data_status,
                        )
                        return (content, metadata) if return_metadata else content
                else:
                    reason = "HTTP status " + str(status_code)
                    counts = None
                retry_after = _retry_after_seconds(response)
                if status_code in (406, 429):
                    retry_statuses.append(status_code)
                    if retry_after is not None:
                        retry_after_values.append(retry_after)
                attempts.append(
                    {
                        "server": true_server_code,
                        "attempt": attempt_number,
                        "http_status": status_code,
                        "reason": reason,
                        "retry_after": retry_after,
                    }
                )
                UI.vprint(
                    1,
                    "        OSM server",
                    true_server_code,
                    "returned invalid data (",
                    reason,
                    ").",
                )
                UI.logprint(
                    "[OSM] query=",
                    query_label,
                    "attempt=",
                    attempt_number,
                    "server=",
                    true_server_code,
                    "status=",
                    status_code,
                    "reason=",
                    reason,
                )
                if status_code != 200:
                    try:
                        UI.vprint(2, "        Server message:", response.text[:200])
                    except Exception:
                        pass
            except requests.RequestException as error:
                attempts.append(
                    {
                        "server": true_server_code,
                        "attempt": attempt_number,
                        "http_status": None,
                        "reason": "request-error",
                        "error": repr(error),
                    }
                )
                retry_statuses.append(None)
                UI.logprint(
                    "[OSM] query=",
                    query_label,
                    "attempt=",
                    attempt_number,
                    "server=",
                    true_server_code,
                    "status=request-error",
                    "error=",
                    error,
                )
                UI.vprint(
                    1,
                    "        OSM server",
                    true_server_code,
                    "request failed:",
                    error,
                )
            except Exception as error:
                attempts.append(
                    {
                        "server": true_server_code,
                        "attempt": attempt_number,
                        "http_status": None,
                        "reason": "unexpected-error",
                        "error": repr(error),
                    }
                )
                retry_statuses.append(None)
                UI.logprint(
                    "[OSM] query=",
                    query_label,
                    "attempt=",
                    attempt_number,
                    "server=",
                    true_server_code,
                    "status=unexpected-error",
                    "error=",
                    error,
                )
                UI.vprint(
                    1,
                    "        OSM server",
                    true_server_code,
                    "request failed:",
                    error,
                )
            if UI.red_flag:
                metadata = {"status": FAILED, "reason": "cancelled", "attempts": attempts}
                return (None, metadata) if return_metadata else 0

        if tentative + 1 >= max(1, int(max_osm_tentatives)):
            break
        delay = _retry_delay(tentative, retry_after_values, retry_statuses)
        UI.vprint(
            1,
            "        All covered Overpass servers failed; new attempt in",
            round(delay, 2),
            "sec...",
        )
        time.sleep(delay)

    metadata = {"status": FAILED, "reason": "all-covered-servers-failed", "attempts": attempts}
    return (None, metadata) if return_metadata else 0

################################################################################
def OSM_to_MultiLineString(
    osm_layer, lat, lon, tags_for_exclusion=None, filter=None
):
    if tags_for_exclusion is None:
        tags_for_exclusion = set()
    multiline = []
    multiline_reject = []
    todo = len(osm_layer.dicosmfirst["w"])
    step = int(todo / 100) + 1
    done = 0
    filtered_segs = 0
    for wayid in sorted(osm_layer.dicosmfirst["w"], key=stable_id_key):
        if done % step == 0:
            UI.progress_bar(1, int(100 * done / todo))
        if (
            tags_for_exclusion
            and wayid in osm_layer.dicosmtags["w"]
            and not set(osm_layer.dicosmtags["w"][wayid].keys()).isdisjoint(
                tags_for_exclusion
            )
        ):
            done += 1
            continue
        way = numpy.round(
            numpy.array(
                [
                    osm_layer.dicosmn[nodeid]
                    for nodeid in osm_layer.dicosmw[wayid]
                ],
                dtype=numpy.float64,
            )
            - numpy.array([[lon, lat]], dtype=numpy.float64),
            7,
        )
        if filter and not filter(way, filtered_segs):
            try:
                multiline_reject.append(geometry.LineString(way))
            except:
                pass
            done += 1
            continue
        try:
            multiline.append(geometry.LineString(way))
            filtered_segs += len(way)
        except:
            pass
        done += 1
    UI.progress_bar(1, 100)
    if not filter:
        return geometry.MultiLineString(multiline)
    else:
        UI.vprint(2, "      Number of filtered segs :", filtered_segs)
        return (
            geometry.MultiLineString(multiline),
            geometry.MultiLineString(multiline_reject),
        )

################################################################################
def OSM_to_MultiPolygon(osm_layer, lat, lon, filter=None):
    multilist = []
    excludelist = []
    todo = len(osm_layer.dicosmfirst["w"]) + len(osm_layer.dicosmfirst["r"])
    step = int(todo / 100) + 1
    done = 0
    for wayid in sorted(osm_layer.dicosmfirst["w"], key=stable_id_key):
        if done % step == 0:
            UI.progress_bar(1, int(100 * done / todo))
        if osm_layer.dicosmw[wayid][0] != osm_layer.dicosmw[wayid][-1]:
            UI.logprint(
                "Non closed way starting at",
                osm_layer.dicosmn[osm_layer.dicosmw[wayid][0]],
                ", skipped.",
            )
            done += 1
            continue
        way = numpy.round(
            numpy.array(
                [
                    osm_layer.dicosmn[nodeid]
                    for nodeid in osm_layer.dicosmw[wayid]
                ],
                dtype=numpy.float64,
            )
            - numpy.array([[lon, lat]], dtype=numpy.float64),
            7,
        )
        try:
            pol = geometry.Polygon(way)
            if not pol.area:
                continue
            if not pol.is_valid:
                UI.logprint(
                    "Invalid OSM way starting at",
                    osm_layer.dicosmn[osm_layer.dicosmw[wayid][0]],
                    ", skipped.",
                )
                done += 1
                continue
        except Exception as e:
            UI.vprint(2, e)
            done += 1
            continue
        if filter and filter(pol, wayid, osm_layer.dicosmtags["w"]):
            excludelist.append(pol)
        else:
            multilist.append(pol)
        done += 1
    for relid in sorted(osm_layer.dicosmfirst["r"], key=stable_id_key):
        if done % step == 0:
            UI.progress_bar(1, int(100 * done / todo))
        try:
            multiout = [
                geometry.Polygon(
                    numpy.round(
                        numpy.array(
                            [osm_layer.dicosmn[nodeid] for nodeid in nodelist],
                            dtype=numpy.float64,
                        )
                        - numpy.array([lon, lat], dtype=numpy.float64),
                        7,
                    )
                )
                for nodelist in osm_layer.dicosmr[relid]["outer"]
            ]
            multiout = ops.unary_union(
                [geom for geom in multiout if geom.is_valid]
            )
            multiin = [
                geometry.Polygon(
                    numpy.round(
                        numpy.array(
                            [osm_layer.dicosmn[nodeid] for nodeid in nodelist],
                            dtype=numpy.float64,
                        )
                        - numpy.array([lon, lat], dtype=numpy.float64),
                        7,
                    )
                )
                for nodelist in osm_layer.dicosmr[relid]["inner"]
            ]
            multiin = ops.unary_union(
                [geom for geom in multiin if geom.is_valid]
            )
        except Exception as e:
            UI.logprint(e)
            done += 1
            continue
        multipol = multiout.difference(multiin)
        if filter and filter(multipol, relid, osm_layer.dicosmtags["r"]):
            targetlist = excludelist
        else:
            targetlist = multilist
        for pol in (
            multipol.geoms
            if (
                "Multi" in multipol.geom_type
                or "Collection" in multipol.geom_type
            )
            else [multipol]
        ):
            if not pol.area:
                done += 1
                continue
            if not pol.is_valid:
                UI.logprint(
                    "Relation",
                    relid,
                    "contains an invalid polygon which was discarded",
                )
                done += 1
                continue
            targetlist.append(pol)
        done += 1
    if filter:
        ret_val = (
            geometry.MultiPolygon(multilist),
            geometry.MultiPolygon(excludelist),
        )
        UI.vprint(
            2,
            "    Total number of geometries:",
            len(ret_val[0].geoms),
            len(ret_val[1].geoms),
        )
    else:
        ret_val = geometry.MultiPolygon(multilist)
        UI.vprint(2, "    Total number of geometries:", len(ret_val.geoms))
    UI.progress_bar(1, 100)
    return ret_val
