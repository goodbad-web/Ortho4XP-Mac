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
from PIL import Image

max_convert_slots = 8
max_download_slots = 8
skip_downloads = False
skip_converts = False


def _ashelper_metal_available(as_helper):
    """Probe ASHelper once before deferring work to the Metal batch path."""
    try:
        result = subprocess.run(
            [as_helper, "--capabilities"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        UI.vprint(1, f"WARNING: Could not probe ASHelper Metal capability: {error}")
        return False

    available = (
        result.returncode == 0
        and "metal_available=true" in (result.stdout or "").splitlines()
    )
    if not available:
        detail = (result.stdout or "").strip()
        if detail:
            UI.vprint(1, "WARNING: ASHelper Metal is unavailable; using CPU conversion.", detail)
        else:
            UI.vprint(1, "WARNING: ASHelper Metal is unavailable; using CPU conversion.")
    return available


def _cpu_fallback_convert_args(convert_list, prepared_input_paths):
    """Build CPU conversion arguments while retaining prepared image work."""
    fallback_convert_list = []
    for item_index, item in enumerate(convert_list):
        prepared_file = (
            prepared_input_paths[item_index]
            if item_index < len(prepared_input_paths)
            else None
        )
        # A direct JPEG still needs the normal CPU-side color/mask
        # preprocessing. Only reuse files that already contain that work.
        if prepared_file:
            item_tile, item_x, item_y, item_z, item_provider = item
            if item_provider in IMG.providers_dict:
                direct_jpeg = os.path.join(
                    FNAMES.jpeg_file_dir_from_attributes(
                        item_tile.lat,
                        item_tile.lon,
                        item_z,
                        IMG.providers_dict[item_provider],
                    ),
                    FNAMES.jpeg_file_name_from_attributes(
                        item_x, item_y, item_z, item_provider
                    ),
                )
                if os.path.abspath(prepared_file) == os.path.abspath(direct_jpeg):
                    prepared_file = None
        if not prepared_file or not os.path.isfile(prepared_file):
            prepared_file = None
        if prepared_file:
            fallback_convert_list.append((*item, "dds", prepared_file))
        else:
            fallback_convert_list.append(item)
    return fallback_convert_list


def _run_cpu_fallback(
    fallback_convert_list, config_data, max_slots, progress
):
    """Run all fallback conversions in the configured CPU pool."""
    cpu_config_data = dict(config_data)
    cpu_config_data.update(
        {
            "use_gpu_acceleration": False,
            "defer_gpu_batch": False,
            "preserve_batch_inputs": True,
        }
    )
    return multiprocessing_pool(
        IMG.convert_texture,
        fallback_convert_list,
        max_slots,
        progress=progress,
        init_func=IMG.init_worker,
        init_args=cpu_config_data,
    )


def _activate_dsf(dsf_tmp_path, dsf_path):
    """Atomically activate a completed DSF while preserving rollback safety."""
    backup_path = dsf_path + ".bak"
    had_existing = os.path.exists(dsf_path)
    if not os.path.isfile(dsf_tmp_path):
        raise FileNotFoundError(dsf_tmp_path)
    if had_existing:
        os.replace(dsf_path, backup_path)
    try:
        os.replace(dsf_tmp_path, dsf_path)
    except Exception:
        if had_existing and not os.path.exists(dsf_path) and os.path.exists(backup_path):
            os.replace(backup_path, dsf_path)
        raise


def _resolve_gpu_batch_mask(tile, til_x_left, til_y_top, zoomlevel, provider_code, png_file_name):
    """Return an exact or materialized mask path for an ASHelper batch task."""
    possible_mask_path = os.path.join(
        tile.build_dir,
        "textures",
        FNAMES.mask_file(til_x_left, til_y_top, zoomlevel, provider_code),
    )
    if os.path.exists(possible_mask_path):
        return possible_mask_path, None

    fallback_mask = MASK.needs_mask(
        tile, til_x_left, til_y_top, zoomlevel, provider_code
    )
    if not fallback_mask:
        return "none", None

    fallback_mask_path = os.path.join(
        UI.Ortho4XP_dir,
        "tmp",
        os.path.splitext(png_file_name)[0] + "_mask.png",
    )
    fallback_mask.convert("L").resize((4096, 4096), Image.BICUBIC).save(
        fallback_mask_path
    )
    return fallback_mask_path, fallback_mask_path

################################################################################
def download_textures(tile, download_queue, convert_queue):
    UI.vprint(1, "-> Opening download queue with", max_download_slots, "workers.")

    def download_task(*texture_attributes):
        if IMG.build_jpeg_ortho(tile, *texture_attributes):
            convert_queue.put((tile, *texture_attributes))
            return 1
        return 0

    dico_dl_progress = {"done": 0, "bar": 2, "message": "Downloading textures"}
    dl_workers = parallel_launch(
        download_task,
        download_queue,
        max_download_slots,
        progress=dico_dl_progress,
    )

    download_success = parallel_join(dl_workers)

    if UI.red_flag:
        UI.vprint(1, "Download process interrupted.")
        return 0

    if not download_success:
        UI.vprint(0, "ERROR: One or more orthophotos could not be downloaded.")
        return 0

    if dico_dl_progress["done"]:
        UI.vprint(1, " *Download of textures completed.")
    return 1

################################################################################
def build_tile(tile):
    if not UI.is_building_all:
        UI.initialize_build_log(tile.build_dir)
    result = 0
    try:
        result = _build_tile(tile)
        return result
    finally:
        # A DSF can be fully written before a later imagery/download stage
        # fails.  Never leave that unactivated artifact behind: the next run
        # must either rebuild it or activate it atomically.
        if not result:
            dsf_tmp_path = os.path.join(
                tile.build_dir,
                "Earth nav data",
                FNAMES.long_latlon(tile.lat, tile.lon) + ".dsf.tmp",
            )
            try:
                os.remove(dsf_tmp_path)
            except OSError:
                pass
        UI.is_working = 0
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

    if not tile.write_to_config():
        UI.exit_message_and_bottom_line("ERROR: Could not save tile configuration.")
        return 0

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
    dsf_state = {"result": 0, "error": None}
    download_state = {"result": 0, "error": None}

    def run_dsf():
        try:
            dsf_state["result"] = DSF.build_dsf(tile, download_queue)
        except Exception as error:
            dsf_state["error"] = error
            UI.vprint(0, "ERROR: DSF worker failed:", error)

    def run_downloads():
        try:
            download_state["result"] = download_textures(
                tile, download_queue, convert_queue
            )
        except Exception as error:
            download_state["error"] = error
            UI.vprint(0, "ERROR: Download worker failed:", error)

    build_dsf_thread = threading.Thread(
        target=run_dsf, name="Ortho4XP-DSF"
    )
    download_thread = threading.Thread(
        target=run_downloads, name="Ortho4XP-downloads"
    )
    build_dsf_thread.start()
    if not skip_downloads:
        download_thread.start()
        download_launched = True
        if not skip_converts:
            dico_conv_progress = {"done": 0, "bar": 3, "message": "Converting DDS textures"}
            convert_launched = True
    build_dsf_thread.join()
    if download_launched:
        for _ in range(max_download_slots):
            download_queue.put("quit")
        download_thread.join()
    if dsf_state["error"] is not None or not dsf_state["result"]:
        UI.exit_message_and_bottom_line("ERROR: DSF construction failed.")
        return 0
    if download_launched and (
        download_state["error"] is not None or not download_state["result"]
    ):
        UI.exit_message_and_bottom_line("ERROR: Texture download failed.")
        return 0
    if convert_launched:
            dds_converter = getattr(tile, 'dds_converter', getattr(UI, 'dds_converter', 'nvcompress'))
            dds_format = getattr(tile, 'dds_format', getattr(UI, 'dds_format', 'BC3'))
            use_gpu = getattr(tile, 'use_gpu_acceleration', getattr(UI, 'use_gpu_acceleration', True))
            as_helper = os.path.join(UI.Ortho4XP_dir, "Utils", "mac", "ASHelper")
            gpu_converter_requested = (
                use_gpu
                and dds_converter == "TextureConverter"
                and "dar" in sys.platform
            )
            gpu_batch_requested = (
                gpu_converter_requested
                and os.path.isfile(as_helper)
                and os.access(as_helper, os.X_OK)
            )
            metal_available = (
                _ashelper_metal_available(as_helper)
                if gpu_converter_requested
                else False
            )
            gpu_batch_enabled = gpu_batch_requested and metal_available
            effective_gpu = use_gpu and (
                not gpu_converter_requested or metal_available
            )
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
                'dds_converter': getattr(tile, 'dds_converter', dds_converter),
                'dds_format': getattr(tile, 'dds_format', dds_format),
                'use_gpu_acceleration': effective_gpu,
                'use_gpu_for_color_filters': getattr(tile, 'use_gpu_for_color_filters', False),
                'is_worker': True
            }
            # Collect conversion arguments from queue
            convert_list = []
            while not convert_queue.empty():
                item = convert_queue.get()
                if item != "quit":
                    convert_list.append(item)

            def can_defer_to_gpu_batch(item):
                _, _, _, _, provider_code = item
                return IMG.can_defer_gpu_batch(provider_code)

            defer_gpu_batch = bool(
                gpu_batch_enabled
                and convert_list
                and all(can_defer_to_gpu_batch(item) for item in convert_list)
            )
            config_data['defer_gpu_batch'] = defer_gpu_batch

            dds_error = IMG.dds_format_support_error(dds_converter, dds_format)
            if dds_error:
                UI.vprint(1, f"ERROR: {dds_error}")
                success_count = 0
                conversion_success = False
            else:
                pool_success = multiprocessing_pool(
                    IMG.convert_texture,
                    convert_list,
                    max_convert_slots,
                    progress=dico_conv_progress,
                    init_func=IMG.init_worker,
                    init_args=config_data
                )
                success_count = len(convert_list) if pool_success else 0
                conversion_success = bool(pool_success)
            
            # GPU Batch DDS Conversion integration for macOS
            if conversion_success and defer_gpu_batch:
                import O4_RAMDisk_Utils
                UI.vprint(1, "-> Executing ultra-fast GPU Batch DDS Conversion via ASHelper...")
                batch_args = []
                temp_files_to_delete = []
                batch_generated_mask_files = []
                prepared_input_paths = [None] * len(convert_list)
                batch_output_specs = []
                batch_attempted = False

                def cleanup_generated_batch_masks():
                    for temp_file in batch_generated_mask_files:
                        try:
                            os.remove(temp_file)
                        except:
                            pass

                def cleanup_batch_outputs():
                    for temp_path, _, _, _ in batch_output_specs:
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass
                
                for item_index, item in enumerate(convert_list):
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
                    
                    if getattr(tile, "use_neural_upscale", False) and os.path.exists(upscaled_tmp):
                        input_path = upscaled_tmp
                        temp_files_to_delete.append(upscaled_tmp)
                    elif getattr(tile, "use_neural_upscale", False) and os.path.exists(tmp_png):
                        input_path = tmp_png
                        temp_files_to_delete.append(tmp_png)
                    elif jpeg_path and IMG._jpeg_file_is_ready(jpeg_path):
                        input_path = jpeg_path
                    else:
                        UI.vprint(1, f"ERROR: Input source image not found for {out_file_name}")
                        conversion_success = False
                        batch_attempted = True
                        break

                    prepared_input_paths[item_index] = input_path
                    
                    mask_path = "none"
                    if input_path == jpeg_path and tile.imprint_masks_to_dds:
                        try:
                            mask_path, generated_mask_path = _resolve_gpu_batch_mask(
                                tile,
                                til_x_left,
                                til_y_top,
                                zoomlevel,
                                provider_code,
                                png_file_name,
                            )
                        except Exception as e:
                            UI.vprint(
                                1,
                                f"ERROR: Could not prepare fallback mask for {out_file_name}: {str(e)}",
                            )
                            conversion_success = False
                            batch_attempted = True
                            break
                        if generated_mask_path:
                            batch_generated_mask_files.append(generated_mask_path)
                        elif mask_path != "none":
                            temp_files_to_delete.append(mask_path)

                    if tile.imprint_masks_to_dds and provider_code in IMG.providers_dict:
                        exact_mask_path = os.path.join(
                            tile.build_dir,
                            "textures",
                            FNAMES.mask_file(
                                til_x_left, til_y_top, zoomlevel, provider_code
                            ),
                        )
                        if (
                            os.path.isfile(exact_mask_path)
                            and exact_mask_path not in temp_files_to_delete
                        ):
                            temp_files_to_delete.append(exact_mask_path)
                    
                    r, g, b = 1.0, 1.0, 1.0
                    contrast, brightness, saturation = 1.0, 0.0, 1.0
                    
                    color_code = "none"
                    if input_path == jpeg_path and provider_code in IMG.providers_dict:
                        color_code = IMG.providers_dict[provider_code].get(
                            "color_filters", "none"
                        )
                        if not IMG.gpu_batch_color_filter_supported(color_code):
                            UI.vprint(
                                1,
                                f"WARNING: Using normal color preprocessing for {out_file_name} ({color_code}).",
                            )
                            conversion_success = False
                            batch_attempted = True
                            break
                        for color_filter in IMG.color_filters_dict.get(color_code, []):
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
                    batch_tmp_path = out_file_path + ".gpu.tmp.dds"
                    try:
                        os.remove(batch_tmp_path)
                    except OSError:
                        pass
                    batch_args.extend([
                        input_path, 
                        mask_path, 
                        str(r), str(g), str(b), 
                        str(contrast), str(brightness), str(saturation), 
                        batch_tmp_path,
                        target_fmt
                    ])
                    batch_output_specs.append(
                        (batch_tmp_path, out_file_path, target_fmt, input_path)
                    )
                
                if conversion_success and batch_args:
                    batch_attempted = True
                    chunk_size = 64
                    for i in range(0, len(batch_args), chunk_size * 10):
                        chunk = batch_args[i:i + chunk_size * 10]
                        cmd = [as_helper, "--convert-batch-v3", "true"] + chunk
                        try:
                            batch_result = subprocess.run(
                                cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                check=False,
                            )
                            ret = batch_result.returncode
                            if batch_result.stdout:
                                output_level = 0 if ret != 0 else 2
                                for line in batch_result.stdout.splitlines():
                                    UI.vprint(output_level, "      " + line)
                            if ret != 0:
                                UI.vprint(1, f"ERROR: GPU Batch DDS conversion failed with return code {ret}")
                                conversion_success = False
                                break
                        except Exception as e:
                            UI.vprint(1, f"ERROR: Execution of GPU Batch DDS conversion failed: {str(e)}")
                            conversion_success = False
                            break
                    if conversion_success:
                        invalid_outputs = []
                        for temp_path, _, target_fmt, input_path in batch_output_specs:
                            try:
                                with Image.open(input_path) as source_image:
                                    expected_dimensions = source_image.size
                            except Exception as error:
                                invalid_outputs.append((temp_path, f"input inspection failed: {error}"))
                                continue
                            dds_valid, dds_error = IMG.validate_dds_file(
                                temp_path,
                                expected_format=target_fmt,
                                expected_dimensions=expected_dimensions,
                                require_mipmaps=True,
                            )
                            if not dds_valid:
                                invalid_outputs.append((temp_path, dds_error))
                        if invalid_outputs:
                            for path, reason in invalid_outputs:
                                UI.vprint(
                                    0,
                                    f"ERROR: GPU batch produced invalid DDS {path}: {reason}",
                                )
                            conversion_success = False
                        else:
                            try:
                                for temp_path, final_path, _, _ in batch_output_specs:
                                    os.replace(temp_path, final_path)
                            except OSError as error:
                                UI.vprint(0, "ERROR: Could not activate GPU batch DDS output:", error)
                                conversion_success = False

                if batch_attempted and not conversion_success:
                    cleanup_batch_outputs()
                    UI.vprint(1, "-> Falling back to CPU DDS conversion via ASHelper...")
                    fallback_convert_list = _cpu_fallback_convert_args(
                        convert_list, prepared_input_paths
                    )
                    for item in convert_list:
                        item_out_name = FNAMES.dds_file_name_from_attributes(
                            item[1], item[2], item[3], item[4]
                        )
                        fallback_tmp_png = os.path.join(
                            UI.Ortho4XP_dir,
                            "tmp",
                            item_out_name.replace("dds", "png"),
                        )
                        if fallback_tmp_png not in temp_files_to_delete:
                            temp_files_to_delete.append(fallback_tmp_png)
                        if item[4] in IMG.providers_dict and item[0].imprint_masks_to_dds:
                            fallback_mask = os.path.join(
                                item[0].build_dir,
                                "textures",
                                FNAMES.mask_file(item[1], item[2], item[3], item[4]),
                            )
                            if os.path.isfile(fallback_mask) and fallback_mask not in temp_files_to_delete:
                                temp_files_to_delete.append(fallback_mask)

                    fallback_progress = {
                        "done": 0,
                        "bar": 3,
                        "message": "CPU fallback DDS conversion",
                    }
                    fallback_success = _run_cpu_fallback(
                        fallback_convert_list,
                        config_data,
                        max_convert_slots,
                        fallback_progress,
                    )
                    success_count = len(convert_list) if fallback_success else 0
                    conversion_success = bool(fallback_success)

                if conversion_success:
                    for temp_file in temp_files_to_delete:
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                    cleanup_generated_batch_masks()

            if not conversion_success:
                UI.lvprint(0, f"WARNING: {len(convert_list) - success_count} textures failed to convert.")
                UI.lvprint(0, "Skipping cleanup to protect existing data.")

            if UI.red_flag:
                UI.vprint(1, "DDS conversion process interrupted.")
            elif dico_conv_progress["done"] >= 1:
                UI.vprint(1, " *DDS conversion of textures completed.")
    if convert_launched and not conversion_success:
        UI.exit_message_and_bottom_line("ERROR: DDS conversion failed.")
        return 0
    if UI.red_flag:
        UI.exit_message_and_bottom_line()
        return 0
    UI.vprint(1, " *Activating DSF file.")
    dsf_file_name = os.path.join(
        tile.build_dir,
        "Earth nav data",
        FNAMES.long_latlon(tile.lat, tile.lon) + ".dsf",
    )
    try:
        _activate_dsf(dsf_file_name + ".tmp", dsf_file_name)
    except Exception as error:
        UI.vprint(0, "ERROR: could not activate DSF file; existing tile was preserved:", error)
        try:
            os.remove(dsf_file_name + ".tmp")
        except OSError:
            pass
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
        UI.is_working = 0
        UI.flush_build_log(tile.build_dir)

def _build_all(tile):
    if not VMAP.build_poly_file(tile) or UI.red_flag:
        UI.exit_message_and_bottom_line("")
        return 0
    if not MESH.build_mesh(tile) or UI.red_flag:
        UI.exit_message_and_bottom_line("")
        return 0
    if not MASK.build_masks(tile) or UI.red_flag:
        UI.exit_message_and_bottom_line("")
        return 0
    if not build_tile(tile) or UI.red_flag:
        UI.exit_message_and_bottom_line("")
        return 0
    if getattr(tile, 'build_overlays_in_all_in_one', False):
        UI.vprint(0, "-> Automatically extracting overlays (All in one)...")
        if not OVL.build_overlay(tile.lat, tile.lon) or UI.red_flag:
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
            if not tile.read_from_config():
                UI.exit_message_and_bottom_line()
                return 0
        if do_osm or do_mesh or do_dsf:
            tile.make_dirs()
        if do_osm:
            if not VMAP.build_poly_file(tile) or UI.red_flag:
                UI.exit_message_and_bottom_line()
                return 0
        if do_mesh:
            if not MESH.build_mesh(tile) or UI.red_flag:
                UI.exit_message_and_bottom_line()
                return 0
        if do_mask:
            if not MASK.build_masks(tile) or UI.red_flag:
                UI.exit_message_and_bottom_line()
                return 0
        if do_dsf:
            if not build_tile(tile) or UI.red_flag:
                UI.exit_message_and_bottom_line()
                return 0
        if do_ovl:
            if not OVL.build_overlay(lat, lon) or UI.red_flag:
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
