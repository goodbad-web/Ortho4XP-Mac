import os
import sys
import subprocess
import time
import shutil
import queue
import threading
import O4_UI_Utils as UI
import O4_File_Names as FNAMES
import O4_Imagery_Utils as IMG
import O4_Vector_Map as VMAP
import O4_Mesh_Utils as MESH
import O4_Mask_Utils as MASK
import O4_DSF_Utils as DSF
import O4_Overlay_Utils as OVL
from O4_Parallel_Utils import parallel_launch, parallel_join, multiprocessing_pool

max_convert_slots = 8
max_download_slots = 8
skip_downloads = False
skip_converts = False

################################################################################
def download_textures(tile, download_queue, convert_queue):
    UI.vprint(1, "-> Opening download queue with", max_download_slots, "workers.")

    def download_task(*texture_attributes):
        if IMG.build_jpeg_ortho(tile, *texture_attributes):
            convert_queue.put((tile, *texture_attributes))
            return 1
        return 0

    dico_dl_progress = {"done": 0, "bar": 2}
    dl_workers = parallel_launch(
        download_task,
        download_queue,
        max_download_slots,
        progress=dico_dl_progress,
    )

    parallel_join(dl_workers)

    if UI.red_flag:
        UI.vprint(1, "Download process interrupted.")
        return 0

    if dico_dl_progress["done"]:
        UI.vprint(1, " *Download of textures completed.")
    return 1

################################################################################
def build_tile(tile):
    if not UI.is_building_all:
        UI.initialize_build_log(tile.build_dir)
    try:
        return _build_tile(tile)
    finally:
        UI.flush_build_log(tile.build_dir)

def _build_tile(tile):
    if UI.is_working:
        return 0
    UI.is_working = 1
    UI.red_flag = False
    UI.logprint(
        "Step 3 for tile lat=", tile.lat, ", lon=", tile.lon, ": starting."
    )
    UI.vprint(
        0,
        "\nStep 3 : Building DSF/Imagery for tile "
        + FNAMES.short_latlon(tile.lat, tile.lon)
        + " : \n--------\n",
    )

    if not os.path.isfile(FNAMES.mesh_file(tile.build_dir, tile.lat, tile.lon)):
        UI.lvprint(
            0, "ERROR: A mesh file must first be constructed for the tile!"
        )
        UI.exit_message_and_bottom_line("")
        return 0

    timer = time.time()

    tile.write_to_config()

    if not IMG.initialize_local_combined_providers_dict(tile):
        UI.exit_message_and_bottom_line("")
        return 0

    try:
        if not os.path.exists(
            os.path.join(
                tile.build_dir,
                "Earth nav data",
                FNAMES.round_latlon(tile.lat, tile.lon),
            )
        ):
            os.makedirs(
                os.path.join(
                    tile.build_dir,
                    "Earth nav data",
                    FNAMES.round_latlon(tile.lat, tile.lon),
                )
            )
        if not os.path.isdir(os.path.join(tile.build_dir, "textures")):
            os.makedirs(os.path.join(tile.build_dir, "textures"))
        if UI.cleaning_level > 1 and not tile.grouped:
            for f in os.listdir(os.path.join(tile.build_dir, "textures")):
                if f[-4:] != ".png":
                    continue
                try:
                    os.remove(os.path.join(tile.build_dir, "textures", f))
                except:
                    pass
        if not tile.grouped:
            try:
                shutil.rmtree(os.path.join(tile.build_dir, "terrain"))
            except:
                pass
        if not os.path.isdir(os.path.join(tile.build_dir, "terrain")):
            os.makedirs(os.path.join(tile.build_dir, "terrain"))
    except Exception as e:
        UI.lvprint(0, "ERROR: Cannot create tile subdirectories.")
        UI.vprint(3, e)
        UI.exit_message_and_bottom_line("")
        return 0

    download_queue = queue.Queue()
    convert_queue = queue.Queue()
    
    download_launched = False
    convert_launched = False
    conversion_success = True

    build_dsf_thread = threading.Thread(
        target=DSF.build_dsf, args=[tile, download_queue]
    )
    download_thread = threading.Thread(
        target=download_textures, args=[tile, download_queue, convert_queue]
    )
    build_dsf_thread.start()
    if not skip_downloads:
        download_thread.start()
        download_launched = True
        if not skip_converts:
            dico_conv_progress = {"done": 0, "bar": 3}
            convert_launched = True
    build_dsf_thread.join()
    if download_launched:
        for _ in range(max_download_slots):
            download_queue.put("quit")
        download_thread.join()
        if convert_launched:
            dds_format = getattr(UI, 'dds_format', 'BC3')
            UI.vprint(
                1,
                "-> Starting multiprocessing pool with",
                max_convert_slots,
                f"workers for DDS conversion (Format: {dds_format}).",
            )
            config_data = {
                'use_magick': IMG.use_magick,
                'use_texture_converter': getattr(IMG, 'use_texture_converter', False),
                'dds_convert_cmd': IMG.dds_convert_cmd,
                'gdal_transl_cmd': IMG.gdal_transl_cmd,
                'gdalwarp_cmd': IMG.gdalwarp_cmd,
                'as_helper_cmd': getattr(IMG, 'as_helper_cmd', None),
                'providers_dict': IMG.providers_dict,
                'local_combined_providers_dict': IMG.local_combined_providers_dict,
                'color_filters_dict': IMG.color_filters_dict,
                'extents_dict': IMG.extents_dict,
                'Ortho4XP_dir': UI.Ortho4XP_dir,
                'verbosity': UI.verbosity,
                'cleaning_level': UI.cleaning_level,
                'use_neural_upscale': getattr(tile, 'use_neural_upscale', False),
                'is_worker': True
            }
            # Collect conversion arguments from queue
            convert_list = []
            while not convert_queue.empty():
                item = convert_queue.get()
                if item != "quit":
                    convert_list.append(item)
            
            success_count = multiprocessing_pool(
                IMG.convert_texture,
                convert_list,
                max_convert_slots,
                progress=dico_conv_progress,
                init_func=IMG.init_worker,
                init_args=config_data
            )

            conversion_success = (success_count == len(convert_list))
            
            # GPU Batch DDS Conversion integration for macOS
            use_gpu = getattr(UI, 'use_gpu_acceleration', True)
            dds_converter = getattr(UI, 'dds_converter', 'nvcompress')
            if conversion_success and use_gpu and dds_converter == "TextureConverter" and "dar" in sys.platform:
                from PIL import Image
                import O4_RAMDisk_Utils
                UI.vprint(1, "-> Executing ultra-fast GPU Batch DDS Conversion via ASHelper...")
                as_helper = os.path.join(UI.Ortho4XP_dir, "Utils", "mac", "ASHelper")
                batch_args = []
                temp_files_to_delete = []
                
                for item in convert_list:
                    tile, til_x_left, til_y_top, zoomlevel, provider_code = item
                    out_file_name = FNAMES.dds_file_name_from_attributes(til_x_left, til_y_top, zoomlevel, provider_code)
                    out_file_path = os.path.join(tile.build_dir, "textures", out_file_name)
                    png_file_name = out_file_name.replace("dds", "png")
                    upscaled_tmp = os.path.join(UI.Ortho4XP_dir, "tmp", out_file_name.replace(".dds", "_upscaled.png"))
                    tmp_png = os.path.join(UI.Ortho4XP_dir, "tmp", png_file_name)
                    
                    if provider_code in IMG.providers_dict:
                        jpeg_file_name = FNAMES.jpeg_file_name_from_attributes(til_x_left, til_y_top, zoomlevel, provider_code)
                        file_dir = FNAMES.jpeg_file_dir_from_attributes(tile.lat, tile.lon, zoomlevel, IMG.providers_dict[provider_code])
                        jpeg_path = os.path.join(file_dir, jpeg_file_name)
                    else:
                        jpeg_path = None
                    
                    if os.path.exists(upscaled_tmp):
                        input_path = upscaled_tmp
                        temp_files_to_delete.append(upscaled_tmp)
                    elif os.path.exists(tmp_png):
                        input_path = tmp_png
                        temp_files_to_delete.append(tmp_png)
                    elif jpeg_path and (os.path.exists(jpeg_path) or O4_RAMDisk_Utils.check_and_restore_cached_image(jpeg_path)):
                        input_path = jpeg_path
                    else:
                        UI.vprint(1, f"ERROR: Input source image not found for {out_file_name}")
                        conversion_success = False
                        break
                    
                    mask_path = "none"
                    if input_path == jpeg_path and tile.imprint_masks_to_dds:
                        possible_mask_path = os.path.join(tile.build_dir, "textures", FNAMES.mask_file(til_x_left, til_y_top, zoomlevel, provider_code))
                        if os.path.exists(possible_mask_path):
                            mask_path = possible_mask_path
                            temp_files_to_delete.append(possible_mask_path)
                    
                    r, g, b = 1.0, 1.0, 1.0
                    contrast, brightness, saturation = 1.0, 0.0, 1.0
                    
                    if input_path == jpeg_path and provider_code in IMG.providers_dict:
                        color_code = IMG.providers_dict[provider_code]["color_filters"]
                        if color_code != "none" and color_code in IMG.color_filters_dict:
                            for color_filter in IMG.color_filters_dict[color_code]:
                                filter_name = color_filter[0]
                                if filter_name == "brightness-contrast":
                                    b_val, c_val = color_filter[1:3]
                                    brightness = b_val / 255.0
                                    contrast = 1.0 + (c_val / 128.0)
                                elif filter_name == "saturation":
                                    s_val = color_filter[1]
                                    saturation = 1.0 + (s_val / 100.0)
                    
                    has_alpha = (mask_path != "none")
                    if not has_alpha and input_path != jpeg_path:
                        try:
                            with Image.open(input_path) as im:
                                if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                                    has_alpha = True
                        except:
                            pass
                    
                    target_fmt = dds_format if (not has_alpha or dds_format == "BC7") else "BC3"
                    batch_args.extend([
                        input_path, 
                        mask_path, 
                        str(r), str(g), str(b), 
                        str(contrast), str(brightness), str(saturation), 
                        out_file_path, 
                        target_fmt
                    ])
                
                if conversion_success and batch_args:
                    chunk_size = 64
                    for i in range(0, len(batch_args), chunk_size * 10):
                        chunk = batch_args[i:i + chunk_size * 10]
                        cmd = [as_helper, "--convert-batch-v3", "true"] + chunk
                        try:
                            ret = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                            if ret != 0:
                                UI.vprint(1, f"ERROR: GPU Batch DDS conversion failed with return code {ret}")
                                conversion_success = False
                                break
                        except Exception as e:
                            UI.vprint(1, f"ERROR: Execution of GPU Batch DDS conversion failed: {str(e)}")
                            conversion_success = False
                            break
                    
                    for temp_file in temp_files_to_delete:
                        try:
                            os.remove(temp_file)
                        except:
                            pass

            if not conversion_success:
                UI.lvprint(0, f"WARNING: {len(convert_list) - success_count} textures failed to convert.")
                UI.lvprint(0, "Skipping cleanup to protect existing data.")

            if UI.red_flag:
                UI.vprint(1, "DDS conversion process interrupted.")
            elif dico_conv_progress["done"] >= 1:
                UI.vprint(1, " *DDS conversion of textures completed.")
    UI.vprint(1, " *Activating DSF file.")
    dsf_file_name = os.path.join(
        tile.build_dir,
        "Earth nav data",
        FNAMES.long_latlon(tile.lat, tile.lon) + ".dsf",
    )
    try:
        os.replace(dsf_file_name + ".tmp", dsf_file_name)
    except:
        UI.vprint(0, "ERROR : could not rename DSF file, tile is not actived.")
    if UI.red_flag:
        UI.exit_message_and_bottom_line()
        return 0
    if UI.cleaning_level > 1:
        try:
            os.remove(FNAMES.alt_file(tile))
        except:
            pass
        try:
            os.remove(FNAMES.input_node_file(tile))
        except:
            pass
        try:
            os.remove(FNAMES.input_poly_file(tile))
        except:
            pass
    if UI.cleaning_level > 2:
        try:
            os.remove(FNAMES.mesh_file(tile.build_dir, tile.lat, tile.lon))
        except:
            pass
        try:
            os.remove(FNAMES.apt_file(tile))
        except:
            pass
    if UI.cleaning_level > 1 and not tile.grouped and conversion_success:
        remove_unwanted_textures(tile)
    try:
        import O4_RAMDisk_Utils
        O4_RAMDisk_Utils.flush_tile_imagery(tile.lat, tile.lon)
    except Exception as e:
        UI.vprint(2, f"[RAMDisk] Warning: Failed to flush tile imagery: {e}")
    UI.timings_and_bottom_line(timer)
    UI.logprint(
        "Step 3 for tile lat=", tile.lat, ", lon=", tile.lon, ": normal exit."
    )
    return 1

################################################################################
def build_all(tile):
    UI.is_building_all = True
    UI.initialize_build_log(tile.build_dir)
    try:
        return _build_all(tile)
    finally:
        UI.is_building_all = False
        UI.flush_build_log(tile.build_dir)

def _build_all(tile):
    VMAP.build_poly_file(tile)
    if UI.red_flag:
        UI.exit_message_and_bottom_line("")
        return 0
    MESH.build_mesh(tile)
    if UI.red_flag:
        UI.exit_message_and_bottom_line("")
        return 0
    MASK.build_masks(tile)
    if UI.red_flag:
        UI.exit_message_and_bottom_line("")
        return 0
    build_tile(tile)
    if UI.red_flag:
        UI.exit_message_and_bottom_line("")
        return 0
    if getattr(tile, 'build_overlays_in_all_in_one', False):
        UI.vprint(0, "-> Automatically extracting overlays (All in one)...")
        OVL.build_overlay(tile.lat, tile.lon)
        if UI.red_flag:
            UI.exit_message_and_bottom_line("")
            return 0
    UI.is_working = 0
    return 1

################################################################################
def build_tile_list(
    tile, list_lat_lon, do_osm, do_mesh, do_mask, do_dsf, do_ovl, do_ptc
):
    if UI.is_working:
        return 0
    UI.red_flag = 0
    timer = time.time()
    UI.lvprint(
        0, "Batch build launched for a number of", len(list_lat_lon), "tiles."
    )
    k = 0
    for (lat, lon) in list_lat_lon:
        k += 1
        UI.vprint(
            1,
            "Dealing with tile ",
            k,
            "/",
            len(list_lat_lon),
            ":",
            FNAMES.short_latlon(lat, lon),
        )
        (tile.lat, tile.lon) = (lat, lon)
        tile.build_dir = FNAMES.build_dir(
            tile.lat, tile.lon, tile.custom_build_dir
        )
        tile.dem = None
        if do_ptc:
            tile.read_from_config()
        if do_osm or do_mesh or do_dsf:
            tile.make_dirs()
        if do_osm:
            VMAP.build_poly_file(tile)
            if UI.red_flag:
                UI.exit_message_and_bottom_line()
                return 0
        if do_mesh:
            MESH.build_mesh(tile)
            if UI.red_flag:
                UI.exit_message_and_bottom_line()
                return 0
        if do_mask:
            MASK.build_masks(tile)
            if UI.red_flag:
                UI.exit_message_and_bottom_line()
                return 0
        if do_dsf:
            build_tile(tile)
            if UI.red_flag:
                UI.exit_message_and_bottom_line()
                return 0
        if do_ovl:
            OVL.build_overlay(lat, lon)
            if UI.red_flag:
                UI.exit_message_and_bottom_line()
                return 0
        try:
            UI.gui.earth_window.canvas.delete(
                UI.gui.earth_window.dico_tiles_todo[(lat, lon)]
            )
            UI.gui.earth_window.dico_tiles_todo.pop((lat, lon), None)
        except:
            pass
    UI.lvprint(
        0, "Batch process completed in", UI.nicer_timer(time.time() - timer)
    )
    return 1

################################################################################
def remove_unwanted_textures(tile):
    texture_list = []
    for f in os.listdir(os.path.join(tile.build_dir, "terrain")):
        if f[-4:] != ".ter":
            continue
        # Extract base texture name by removing suffixes
        base_name = f[:-4].replace("_water", "").replace("_sea", "").replace("_overlay", "")
        texture_list.append(base_name + ".dds")
    for f in os.listdir(os.path.join(tile.build_dir, "textures")):
        if f[-4:] != ".dds":
            continue
        if f not in texture_list:
            print("Removing obsolete texture", f)
            try:
                os.remove(os.path.join(tile.build_dir, "textures", f))
            except:
                pass
