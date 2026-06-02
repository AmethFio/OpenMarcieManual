#!/usr/bin/env bash
# ============================================================
# Download script for BikeExperiment dataset (DFKI Open-MARCIE)
# Auto-discovers all available volumes (Vol1, Vol2, ...) and
# downloads for each volume:
#   - Wearables/EgoVideo<VolN>.zip
#   - HardLabels/vol<N>_VideoChestLabelsSec.csv
# ============================================================

BASE_URL="https://projects.dfki.uni-kl.de/open-marcie/BikeExperiment"
USER="reviewer"
PASS="1234"
MAX_VOL=99   # safety upper bound

download_file() {
  local url="$1"
  local local_path="$2"

  local local_dir
  local_dir=$(dirname "$local_path")
  mkdir -p "$local_dir"

  echo "  -> Downloading: $url"
  HTTP_CODE=$(curl -u "${USER}:${PASS}" \
                   -L \
                   --retry 3 \
                   --retry-delay 5 \
                   --progress-bar \
                   -w "%{http_code}" \
                   -o "${local_path}" \
                   "${url}")

  if [ "$HTTP_CODE" -eq 200 ]; then
    echo "  [OK] Saved: ${local_path}"
    return 0
  else
    echo "  [SKIP] HTTP ${HTTP_CODE} for ${url}"
    rm -f "${local_path}"   # remove empty/error file
    return 1
  fi
}

echo "============================================================"
echo " BikeExperiment Dataset Downloader"
echo " Base URL : ${BASE_URL}"
echo "============================================================"

found_any=0

for i in $(seq 1 $MAX_VOL); do
  VOL="Vol${i}"
  n=$(printf "%d" "$i")   # plain number, e.g. 1, 2, 3

  # Build URLs
  EGO_URL="${BASE_URL}/${VOL}/Wearables/EgoVideo${VOL}.zip"
  CSV_URL="${BASE_URL}/${VOL}/HardLabels/vol${n}_VideoChestLabelsSec.csv"

  # Probe: check if EgoVideo zip exists (HEAD request)
  PROBE=$(curl -u "${USER}:${PASS}" \
               -L --silent --head \
               -w "%{http_code}" -o /dev/null \
               "${EGO_URL}")

  if [ "$PROBE" -ne 200 ]; then
    echo ""
    echo "------------------------------------------------------------"
    echo "${VOL} not found (HTTP ${PROBE}) — stopping."
    echo "------------------------------------------------------------"
    break
  fi

  echo ""
  echo "============================================================"
  echo " Processing ${VOL}"
  echo "============================================================"
  found_any=1

  # Download EgoVideo zip
  download_file "${EGO_URL}" "${VOL}/Wearables/EgoVideo${VOL}.zip"

  # Download CSV labels
  download_file "${CSV_URL}" "${VOL}/HardLabels/vol${n}_VideoChestLabelsSec.csv"

done

echo ""
echo "============================================================"
if [ "$found_any" -eq 1 ]; then
  echo "All available volumes downloaded."
else
  echo "No volumes found. Check credentials or base URL."
fi
echo "============================================================"
