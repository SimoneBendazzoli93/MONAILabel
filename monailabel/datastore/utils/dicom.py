# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

from dicomweb_client import DICOMwebClient
from pydicom.dataset import Dataset
from pydicom.filereader import dcmread

from monailabel.utils.others.generic import md5_digest, run_command
import math

logger = logging.getLogger(__name__)

def normalize_PET_to_SUV_BW(slice):
    corrected_image = slice[0x0028, 0x0051].value
    decay_correction = slice[0x0054, 0x1102].value
    units = slice[0x0054, 0x1001].value

    series_date = slice.SeriesDate
    acquisition_date = slice.AcquisitionDate
    series_time = slice.SeriesTime
    acquisition_time = slice.AcquisitionTime
    half_life = slice.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
    weight = slice.PatientWeight

    if "ATTN" in corrected_image and "DECY" in corrected_image and decay_correction == "START":
        if units == "BQML":
            if series_time <= acquisition_time and series_date <= acquisition_date:
                scan_date = series_date
                scan_time = series_time
            else:
                scan_date = acquisition_date
                scan_time = acquisition_time
            # if not RadiopharmaceuticalStartTime in ds.RadiopharmaceuticalInformationSequence[0]:
            #    ...
            # else:
            start_time = slice.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
            start_date = scan_date

            scan_time = str(round(float(scan_time)))
            str_scan_time = time.strptime(scan_date + scan_time, "%Y%m%d%H%M%S")

            start_time = str(round(float(start_time)))

            str_start_time = time.strptime(start_date + start_time, "%Y%m%d%H%M%S")

            decay_time = time.mktime(str_scan_time) - time.mktime(str_start_time)

            injected_dose = slice.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
            decayed_dose = injected_dose * math.pow(2, -decay_time / half_life)

            SUB_BW_scale_factor = (weight * 1000) / decayed_dose

    return SUB_BW_scale_factor

def generate_key(patient_id: str, study_id: str, series_id: str):
    return md5_digest(f"{patient_id}+{study_id}+{series_id}")


def get_scu(query, output_dir, query_level="SERIES", host="127.0.0.1", port="4242", aet="MONAILABEL"):
    start = time.time()
    field = "StudyInstanceUID" if query_level == "STUDIES" else "SeriesInstanceUID"
    run_command(
        "python",
        [
            "-m",
            "pynetdicom",
            "getscu",
            host,
            port,
            "-P",
            "-k",
            f"0008,0052={query_level}",
            "-k",
            f"{field}={query}",
            "-aet",
            aet,
            "-q",
            "-od",
            output_dir,
        ],
    )
    logger.info(f"Time to run GET-SCU: {time.time() - start} (sec)")


def store_scu(input_file, host="127.0.0.1", port="4242", aet="MONAILABEL"):
    start = time.time()
    input_files = input_file if isinstance(input_file, list) else [input_file]
    for i in input_files:
        run_command("python", ["-m", "pynetdicom", "storescu", host, port, "-aet", aet, i])
    logger.info(f"Time to run STORE-SCU: {time.time() - start} (sec)")


def dicom_web_download_series(study_id, series_id, save_dir, client: DICOMwebClient, frame_fetch=False):
    start = time.time()

    # Limitation for DICOMWeb Client as it needs StudyInstanceUID to fetch series
    if not study_id:
        meta = Dataset.from_json(
            [
                series
                for series in client.search_for_series(search_filters={"SeriesInstanceUID": series_id})
                if series["0020000E"]["Value"] == [series_id]
            ][0]
        )
        study_id = str(meta["StudyInstanceUID"].value)

    os.makedirs(save_dir, exist_ok=True)
    # Retrieve all series of a study if series_id is None
    if series_id is None:
        series_list = client.search_for_series(search_filters={"StudyInstanceUID": study_id})
        
        series_list = [Dataset.from_json(s)["SeriesInstanceUID"].value for s in series_list]
        print(series_list)
        logger.info(f"Found {len(series_list)} series in study {study_id}")
    else:
        series_list = [series_id]
    for series_id in series_list:
        print(f"++ Downloading Series: {series_id}")
        os.makedirs(os.path.join(save_dir, series_id), exist_ok=True)
        if not frame_fetch:
            instances = client.retrieve_series(study_id, series_id)
            for instance in instances:
                instance_id = str(instance["SOPInstanceUID"].value)
                file_name = os.path.join(save_dir, series_id, f"{instance_id}.dcm")
                # Check the Modality of the DICOM instance
                modality = getattr(instance, "Modality", None)
                logger.info(f"Modality for instance {instance_id}: {modality}")
                if modality == "PT":
                    suv_factor = normalize_PET_to_SUV_BW(instance)
                    instance.RescaleSlope = suv_factor * instance.RescaleSlope
                    logger.info(f"Normalized SUV BW for instance {instance_id}: {suv_factor}")
                instance.save_as(file_name)
        else:
            # TODO:: This logic (combining meta+pixeldata) needs improvement
            def save_from_frame(m):
                d = Dataset.from_json(m)
                instance_id = str(d["SOPInstanceUID"].value)

                # Hack to merge Info + RawData
                d.is_little_endian = True
                d.is_implicit_VR = True
                d.PixelData = client.retrieve_instance_frames(
                    study_instance_uid=study_id,
                    series_instance_uid=series_id,
                    sop_instance_uid=instance_id,
                    frame_numbers=[1],
                )[0]

                file_name = os.path.join(save_dir, series_id, f"{instance_id}.dcm")
                logger.info(f"++ Saved {os.path.basename(file_name)}")
                modality = getattr(d, "Modality", None)
                logger.info(f"Modality for instance {instance_id}: {modality}")
                if modality == "PT":
                    suv_factor = normalize_PET_to_SUV_BW(d)
                    d.RescaleSlope = suv_factor * d.RescaleSlope
                    logger.info(f"Normalized SUV BW for instance {instance_id}: {suv_factor}")
                d.save_as(file_name)

            meta_list = client.retrieve_series_metadata(study_id, series_id)
            logger.info(f"++ Saving DCM into: {save_dir}/{series_id}/")
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="DICOMFetch") as executor:
                executor.map(save_from_frame, meta_list)

    logger.info(f"Time to download: {time.time() - start} (sec)")


def dicom_web_upload_dcm(input_file, client: DICOMwebClient):
    start = time.time()
    dataset = dcmread(input_file)
    result = client.store_instances([dataset])

    url = ""
    for elm in result.iterall():
        s = str(elm.value)
        logger.info(f"{s}")
        if "/series/" in s:
            url = s
            break

    series_id = url.split("/series/")[1].split("/")[0] if url else ""
    logger.info(f"Series Instance UID: {series_id}")

    logger.info(f"Time to upload: {time.time() - start} (sec)")
    return series_id


if __name__ == "__main__":
    import shutil

    from monailabel.datastore.dicom import DICOMwebClientX

    client = DICOMwebClientX(
        url="https://d1l7y4hjkxnyal.cloudfront.net",
        session=None,
        qido_url_prefix="output",
        wado_url_prefix="output",
        stow_url_prefix="output",
    )

    study_id = "1.2.840.113654.2.55.68425808326883186792123057288612355322"
    series_id = "1.2.840.113654.2.55.257926562693607663865369179341285235858"

    save_dir = "/local/sachi/Data/dicom"
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)

    dicom_web_download_series(study_id, series_id, save_dir, client, True)
