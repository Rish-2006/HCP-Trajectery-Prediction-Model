import os
import json
import math
import glob
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union


class NuScenesMapWrapper:
    """
    Wrapper for nuScenes Map queries. Provides lane centerlines, crosswalks,
    and drivable area within a given ego-radius.

    Tries to load the *real* map using nuscenes-devkit's NuScenesMap (reads
    the vector layers from <map_dir>/maps/expansion/<location>.json, which is
    what the official map-expansion .zip is meant to populate). If that
    fails for any reason (package not installed, expansion pack not
    extracted yet, folder layout doesn't match, location name not found,
    etc.) it falls back to the previous procedurally-generated placeholder
    map, so training never breaks because of a missing/misplaced file.
    """

    def __init__(self, map_dir, location="singapore-onenorth",
                 cache_resolution=20.0, cache_max_entries=5000):
        """
        cache_resolution : ego positions are rounded to this many meters
                            before querying, so nearby positions reuse the
                            same cached map-query result instead of
                            re-running the (expensive) real map query from
                            scratch. This matters a lot in practice: a full
                            nuScenes scene reuses the same handful of ego
                            positions across many different agent-track
                            slices, so most calls are near-duplicates.
        cache_max_entries : caps cache memory growth; a large trainval run
                            has many unique scenes, so this bounds how much
                            gets cached rather than growing unbounded.
        """
        self.map_dir = map_dir
        self.location = location
        self.is_mock = True
        self.nusc_map = None
        self._cache = {}
        self._cache_resolution = cache_resolution
        self._cache_max_entries = cache_max_entries
        self._load_real_map()

    def _load_real_map(self):
        try:
            from nuscenes.map_expansion.map_api import NuScenesMap
        except ImportError:
            print("nuscenes-devkit not importable — using procedural mock map.")
            return

        # The devkit expects <dataroot>/maps/expansion/<map_name>.json.
        # Map-expansion zips extract with varying internal layouts depending
        # on how/when they were downloaded (nested under maps/expansion/,
        # flat under expansion/, flat at the root, etc.) — rather than
        # trying to infer dataroot by counting directory levels above
        # wherever the file happens to be found (fragile, breaks silently
        # on layouts that don't match what was assumed), always copy
        # whatever file is found into the exact path the devkit expects,
        # then point dataroot at self.map_dir directly. Idempotent — skips
        # the copy if it's already sitting in the right place.
        target_dir  = os.path.join(self.map_dir, "maps", "expansion")
        target_path = os.path.join(target_dir, f"{self.location}.json")

        if not os.path.exists(target_path):
            hits = glob.glob(os.path.join(self.map_dir, "**", f"{self.location}.json"), recursive=True)
            if not hits:
                print(f"No {self.location}.json map-expansion file found under {self.map_dir} "
                      f"— using procedural mock map until it's extracted there.")
                return
            try:
                import shutil
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy(hits[0], target_path)
            except Exception as e:
                print(f"Found a map-expansion file at {hits[0]} but couldn't stage it "
                      f"into the layout the devkit expects ({e}) — using procedural mock map.")
                return

        try:
            self.nusc_map = NuScenesMap(dataroot=self.map_dir, map_name=self.location)
            self.is_mock = False
            print(f"Loaded real nuScenes map for '{self.location}' from {self.map_dir}.")
        except Exception as e:
            print(f"Found map-expansion files but failed to load real map ({e}) — "
                  f"using procedural mock map.")
            self.nusc_map = None
            self.is_mock = True

    # ------------------------------------------------------------------
    def get_map_elements(self, ego_x, ego_y, radius=500.0):
        """
        Returns lane centerlines, crosswalk polygons, and drivable area.
        Cached by (rounded ego position, radius) — see __init__ docstring.
        """
        cache_key = (
            round(ego_x / self._cache_resolution),
            round(ego_y / self._cache_resolution),
            radius,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if not self.is_mock and self.nusc_map is not None:
            try:
                result = self._get_real_map_elements(ego_x, ego_y, radius)
            except Exception as e:
                print(f"Real map query failed ({e}) — falling back to procedural mock "
                      f"map for this call.")
                result = self._get_mock_map_elements(ego_x, ego_y, radius)
        else:
            result = self._get_mock_map_elements(ego_x, ego_y, radius)

        if len(self._cache) < self._cache_max_entries:
            self._cache[cache_key] = result
        return result

    def _get_real_map_elements(self, ego_x, ego_y, radius):
        nm = self.nusc_map
        records = nm.get_records_in_radius(
            ego_x, ego_y, radius, layer_names=["lane", "lane_connector", "ped_crossing", "drivable_area"]
        )

        # --- lane centerlines (lane + lane_connector share the same discretizer) ---
        lane_tokens = records.get("lane", []) + records.get("lane_connector", [])
        lanes = []
        if lane_tokens:
            discretized = nm.discretize_lanes(lane_tokens, resolution_meters=1.0)
            for token, pts in discretized.items():
                if len(pts) >= 2:
                    lanes.append(LineString([(p[0], p[1]) for p in pts]))

        # --- crosswalks: each ped_crossing record has a polygon_token ---
        crosswalks = []
        for tok in records.get("ped_crossing", []):
            try:
                rec = nm.get("ped_crossing", tok)
                poly = nm.extract_polygon(rec["polygon_token"])
                if poly.is_valid and not poly.is_empty:
                    crosswalks.append(poly)
            except Exception:
                continue

        # --- drivable area: union every polygon under every matched record ---
        drivable_polys = []
        for tok in records.get("drivable_area", []):
            try:
                rec = nm.get("drivable_area", tok)
                for poly_tok in rec.get("polygon_tokens", []):
                    poly = nm.extract_polygon(poly_tok)
                    if poly.is_valid and not poly.is_empty:
                        drivable_polys.append(poly)
            except Exception:
                continue

        if drivable_polys:
            drivable_area = unary_union(drivable_polys)
        else:
            # No drivable_area record within radius — fall back to a coarse
            # bounding envelope around the ego position rather than leaving
            # it empty, so downstream code (which expects one polygon)
            # doesn't have to special-case "no drivable area".
            drivable_area = Polygon([
                (ego_x - radius, ego_y - radius), (ego_x + radius, ego_y - radius),
                (ego_x + radius, ego_y + radius), (ego_x - radius, ego_y + radius),
            ])

        return {"lanes": lanes, "crosswalks": crosswalks, "drivable_area": drivable_area}

    def _get_mock_map_elements(self, ego_x, ego_y, radius=500.0):
        # Procedurally generate map lanes, crosswalks, and drivable areas in
        # ego-centric coords. Unchanged fallback behaviour from before.
        lanes = []
        crosswalks = []

        for offset in [-3.5, 0.0, 3.5]:
            lane_pts = []
            for y in np.arange(-radius, radius, 10.0):
                x = offset + 10.0 * math.sin(y / 150.0)
                lane_pts.append((x + ego_x, y + ego_y))
            lanes.append(LineString(lane_pts))

        for offset in [-3.5, 0.0, 3.5]:
            lane_pts = []
            for x in np.arange(-radius, radius, 10.0):
                y = offset + 10.0 * math.cos(x / 150.0)
                lane_pts.append((x + ego_x, y + ego_y))
            lanes.append(LineString(lane_pts))

        for x_val in [-100.0, 100.0]:
            poly = Polygon([
                (x_val - 5.0 + ego_x, -20.0 + ego_y),
                (x_val + 5.0 + ego_x, -20.0 + ego_y),
                (x_val + 5.0 + ego_x, 20.0 + ego_y),
                (x_val - 5.0 + ego_x, 20.0 + ego_y)
            ])
            crosswalks.append(poly)

        drivable_coords = [
            (-50.0 + ego_x, -radius + ego_y),
            (50.0 + ego_x, -radius + ego_y),
            (radius + ego_x, -50.0 + ego_y),
            (radius + ego_x, 50.0 + ego_y),
            (50.0 + ego_x, radius + ego_y),
            (-50.0 + ego_x, radius + ego_y),
            (-radius + ego_x, 50.0 + ego_y),
            (-radius + ego_x, -50.0 + ego_y),
        ]
        drivable_area = Polygon(drivable_coords)

        return {"lanes": lanes, "crosswalks": crosswalks, "drivable_area": drivable_area}


# ---------------------------------------------------------------------------
# Sensor (camera / LiDAR) file loaders
# ---------------------------------------------------------------------------
# These read the raw blob files nuScenes ships in v1.0-trainvalNN_blobs.tgz.
# NOTE: nothing in MTRMotionTransformer currently consumes images or point
# clouds as model input — the transformer's tokenizer only takes agent
# tracks + map polylines. These loaders make the blob data indexed and
# ready to use; actually learning from raw pixels/points needs a new
# encoder branch feeding into the fusion stage, which is a separate,
# larger architecture change.

def load_camera_image(data_dir, rel_path):
    """Load a camera keyframe as a PIL RGB image. rel_path is the
    'filename' field from sample_data.json, e.g. 'samples/CAM_FRONT/xxx.jpg'.
    Returns None if the file isn't present (e.g. that blob part wasn't
    downloaded)."""
    full_path = os.path.join(data_dir, rel_path)
    if not os.path.exists(full_path):
        return None
    from PIL import Image
    return Image.open(full_path).convert("RGB")


def load_lidar_points(data_dir, rel_path):
    """Load a LIDAR_TOP sweep. nuScenes stores these as raw float32 binary
    files with 5 columns per point: x, y, z, intensity, ring_index (in the
    sensor's own frame — not yet transformed to ego/world frame).
    Returns an (N, 5) numpy array, or None if the file isn't present."""
    full_path = os.path.join(data_dir, rel_path)
    if not os.path.exists(full_path):
        return None
    points = np.fromfile(full_path, dtype=np.float32)
    if points.size % 5 != 0:
        print(f"Warning: {rel_path} isn't a multiple of 5 floats — "
              f"may not be a standard nuScenes LIDAR_TOP sweep.")
        return None
    return points.reshape(-1, 5)


class NuScenesParser:
    def __init__(self, data_dir, version=None, scene_filter=None):
        """
        version: name of the metadata folder to load from data_dir, e.g.
        "v1.0-trainval" (the real full split) or "v1.0-mini" (small dev
        split). If left as None (default), auto-detects by checking which
        of the two actually exists under data_dir — preferring
        v1.0-trainval if both are present — instead of assuming one
        hardcoded value.
        scene_filter: optional set/list of scene names (e.g. {"scene-0003",
        "scene-0012", ...}) to restrict processing to. Use this to build a
        train-only or val-only subset — e.g. via
        nuscenes.utils.splits.create_splits_scenes()['val'] for the
        official nuScenes val split. None (default) processes every scene.
        """
        self.data_dir = data_dir
        if version is None:
            version = self._detect_version(data_dir)
        self.meta_dir = os.path.join(data_dir, version)
        self.scene_filter = set(scene_filter) if scene_filter is not None else None

    @staticmethod
    def _detect_version(data_dir):
        if os.path.isdir(os.path.join(data_dir, "v1.0-trainval")):
            return "v1.0-trainval"
        if os.path.isdir(os.path.join(data_dir, "v1.0-mini")):
            return "v1.0-mini"
        return "v1.0-trainval"

    def load_table(self, table_name):
        path = os.path.join(self.meta_dir, f"{table_name}.json")
        if not os.path.exists(path):
            return []
        with open(path, 'r') as f:
            return json.load(f)

    def _build_sensor_index(self, samples):
        """
        Maps sample_token -> {channel: relative_filename} for keyframe
        sensor data (camera + LIDAR), using sample_data.json joined through
        calibrated_sensor.json / sensor.json to get each record's channel
        name (e.g. 'CAM_FRONT', 'LIDAR_TOP'). Returns {} if sample_data
        wasn't downloaded/extracted (e.g. only the metadata archive is
        present so far, no blobs) — that's a normal, expected state, not
        an error.

        sample_data.json is by far the largest table (every keyframe AND
        every intermediate sweep, for every sensor, for the whole dataset —
        multiple millions of rows on the full trainval split). Loading it
        whole via json.load() before filtering can use several GB of RAM
        just for that one table, risking an out-of-memory crash on a
        constrained environment like Colab's free tier. So this streams the
        file with ijson instead when available, filtering to keyframes as
        it goes rather than ever holding the full unfiltered list in memory.
        """
        sample_data_path = os.path.join(self.meta_dir, "sample_data.json")
        if not os.path.exists(sample_data_path):
            return {}

        calibrated_sensors = self.load_table("calibrated_sensor")
        sensors = self.load_table("sensor")
        if not calibrated_sensors or not sensors:
            return {}

        sensor_by_tok = {s["token"]: s["channel"] for s in sensors}
        channel_by_cs_tok = {
            cs["token"]: sensor_by_tok.get(cs["sensor_token"])
            for cs in calibrated_sensors
        }

        try:
            import ijson
            print("  Streaming sample_data.json with ijson (keeps memory bounded "
                  "regardless of file size — the multi-million-row full-trainval "
                  "table won't be held whole in RAM)...")
            index = {}
            count = 0
            with open(sample_data_path, "rb") as f:
                for sd in ijson.items(f, "item"):
                    count += 1
                    if count % 500000 == 0:
                        print(f"    ...{count} sample_data entries scanned "
                              f"({len(index)} samples indexed so far)")
                    if not sd.get("is_key_frame", False):
                        continue
                    channel = channel_by_cs_tok.get(sd["calibrated_sensor_token"])
                    if channel is None:
                        continue
                    index.setdefault(sd["sample_token"], {})[channel] = sd["filename"]
            print(f"  Sensor index built: {count} sample_data entries scanned, "
                  f"{len(index)} samples indexed.")
            return index
        except ImportError:
            print("  ijson not installed — falling back to loading sample_data.json "
                  "whole (uses more memory; pip install ijson to avoid this).")
            return self._build_sensor_index_fallback(sample_data_path, channel_by_cs_tok)

    def _build_sensor_index_fallback(self, sample_data_path, channel_by_cs_tok):
        """Non-streaming fallback for _build_sensor_index when ijson isn't
        available — loads the whole table into memory at once."""
        with open(sample_data_path, "r") as f:
            sample_data = json.load(f)
        print(f"  Loaded {len(sample_data)} sample_data entries.")

        index = {}
        total = len(sample_data)
        for i, sd in enumerate(sample_data):
            if i > 0 and i % 500000 == 0:
                print(f"    ...{i}/{total} sample_data entries scanned "
                      f"({len(index)} samples indexed so far)")
            if not sd.get("is_key_frame", False):
                continue
            channel = channel_by_cs_tok.get(sd["calibrated_sensor_token"])
            if channel is None:
                continue
            index.setdefault(sd["sample_token"], {})[channel] = sd["filename"]
        print(f"  Sensor index built: {len(index)} samples indexed.")
        return index

    def process_dataset(self):
        """
        Parses agent tracks and maps.
        Extracts 2s history (5 steps at 2Hz) and 6s future (12 steps at 2Hz).
        """
        scenes = self.load_table("scene")
        samples = self.load_table("sample")
        ego_poses = self.load_table("ego_pose")
        annotations = self.load_table("sample_annotation")
        instances = self.load_table("instance")
        categories = self.load_table("category")

        if not scenes:
            print("No nuScenes meta tables found or dataset is empty. Using mock data.")
            return []

        if self.scene_filter is not None:
            allowed_scene_tokens = {s["token"] for s in scenes if s["name"] in self.scene_filter}
            # Filter samples and ego_poses together, preserving their existing
            # positional alignment (ego_pose_dict below pairs them by index).
            keep_mask = [s["scene_token"] in allowed_scene_tokens for s in samples]
            samples   = [s for s, keep in zip(samples, keep_mask) if keep]
            ego_poses = [e for e, keep in zip(ego_poses, keep_mask) if keep]
            scenes    = [s for s in scenes if s["token"] in allowed_scene_tokens]
            allowed_sample_tokens = {s["token"] for s in samples}
            annotations = [a for a in annotations if a["sample_token"] in allowed_sample_tokens]
            print(f"Scene filter applied: restricted to {len(scenes)} scene(s) "
                  f"({len(self.scene_filter)} requested) -> {len(samples)} samples, "
                  f"{len(annotations)} annotations remain.")

        print(f"Loading nuScenes tables: {len(scenes)} scenes, {len(samples)} samples, "
              f"{len(annotations)} annotations, {len(instances)} instances...")

        sample_dict = {s["token"]: s for s in samples}
        ego_pose_dict = {s["token"]: e for s, e in zip(samples, ego_poses)}
        cat_dict = {c["token"]: c["name"] for c in categories}
        inst_dict = {i["token"]: i for i in instances}
        print("Building sensor file index (sample_data/calibrated_sensor/sensor)...")
        sensor_index = self._build_sensor_index(samples)

        ann_by_inst = {}
        for ann in annotations:
            inst_tok = ann["instance_token"]
            if inst_tok not in ann_by_inst:
                ann_by_inst[inst_tok] = []
            ann_by_inst[inst_tok].append(ann)

        print(f"Processing {len(ann_by_inst)} instance tracks into trajectory slices...")
        processed_tracks = []
        num_instances = len(ann_by_inst)

        for processed_count, (inst_tok, anns) in enumerate(ann_by_inst.items()):
            if processed_count > 0 and processed_count % 5000 == 0:
                print(f"  ...{processed_count}/{num_instances} instances processed "
                      f"({len(processed_tracks)} trajectory slices so far)")
            anns = sorted(anns, key=lambda a: sample_dict[a["sample_token"]]["timestamp"])
            if len(anns) < 17:
                continue

            inst = inst_dict[inst_tok]
            cat_name = cat_dict[inst["category_token"]]

            states = []
            for i in range(len(anns)):
                ann = anns[i]
                sample = sample_dict[ann["sample_token"]]
                t_us = sample["timestamp"]
                x, y, z = ann["translation"]
                qw, qx, qy, qz = ann["rotation"]

                heading = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

                states.append({
                    "timestamp_us": t_us,
                    "x": x,
                    "y": y,
                    "heading_rad": heading,
                    "width": ann["size"][0],
                    "length": ann["size"][1],
                    # Relative paths into samples/CAM_FRONT/..., samples/LIDAR_TOP/...
                    # (empty dict if sample_data/blobs weren't extracted).
                    # Use load_camera_image()/load_lidar_points() to read them.
                    "sensor_files": sensor_index.get(ann["sample_token"], {}),
                })

            for i in range(len(states)):
                t_curr = states[i]["timestamp_us"] / 1e6
                x_curr = states[i]["x"]
                y_curr = states[i]["y"]

                if i > 0 and i < len(states) - 1:
                    t_prev = states[i-1]["timestamp_us"] / 1e6
                    t_next = states[i+1]["timestamp_us"] / 1e6

                    vx = (states[i+1]["x"] - states[i-1]["x"]) / (t_next - t_prev)
                    vy = (states[i+1]["y"] - states[i-1]["y"]) / (t_next - t_prev)

                    ax = (states[i+1]["x"] - 2 * x_curr + states[i-1]["x"]) / ((t_next - t_prev) / 2) ** 2
                    ay = (states[i+1]["y"] - 2 * y_curr + states[i-1]["y"]) / ((t_next - t_prev) / 2) ** 2
                elif i == 0:
                    t_next = states[i+1]["timestamp_us"] / 1e6
                    vx = (states[i+1]["x"] - x_curr) / (t_next - t_curr)
                    vy = (states[i+1]["y"] - y_curr) / (t_next - t_curr)
                    ax, ay = 0.0, 0.0
                else:
                    t_prev = states[i-1]["timestamp_us"] / 1e6
                    vx = (x_curr - states[i-1]["x"]) / (t_curr - t_prev)
                    vy = (y_curr - states[i-1]["y"]) / (t_curr - t_prev)
                    ax, ay = 0.0, 0.0

                states[i]["vx"] = vx
                states[i]["vy"] = vy
                states[i]["ax"] = ax
                states[i]["ay"] = ay
                states[i]["track_id"] = inst_tok
                states[i]["category"] = cat_name

            for pivot in range(4, len(states) - 12):
                history = states[pivot-4: pivot+1]
                future = states[pivot+1: pivot+13]

                processed_tracks.append({
                    "track_id": inst_tok,
                    "category": cat_name,
                    "history": history,
                    "future": future,
                    "ego_pose": ego_pose_dict[anns[pivot]["sample_token"]]
                })

        return processed_tracks


if __name__ == "__main__":
    parser = NuScenesParser(os.path.join(os.path.dirname(__file__), "nuscenes"))
    tracks = parser.process_dataset()
    print(f"Processed {len(tracks)} trajectory slices from nuScenes.")
    if tracks:
        sample_sensor_files = tracks[0]["history"][-1].get("sensor_files", {})
        print(f"Sensor channels indexed for first track's latest history frame: "
              f"{list(sample_sensor_files.keys())}")