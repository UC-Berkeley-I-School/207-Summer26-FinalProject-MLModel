from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import pyarrow.parquet as pq
import requests
from bs4 import BeautifulSoup


BTS_BASE_URL = "https://transtats.bts.gov/PREZIP"
BTS_PREFIX = "On_Time_Reporting_Carrier_On_Time_Performance_1987_present"
NOAA_BASE_URL = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly"
FAA_DOWNLOAD_PAGE = (
    "https://www.faa.gov/licenses_certificates/aircraft_certification/"
    "aircraft_registry/releasable_aircraft_download"
)

BTS_EXPORT_COLUMNS = [
    "Year", "Month", "FlightDate", "Reporting_Airline",
    "DOT_ID_Reporting_Airline", "IATA_CODE_Reporting_Airline", "Tail_Number",
    "Flight_Number_Reporting_Airline", "OriginAirportID", "Origin",
    "OriginCityName", "OriginState", "DestAirportID", "Dest", "DestCityName",
    "DestState", "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes",
    "DepDel15", "DepartureDelayGroups", "TaxiOut", "WheelsOff", "WheelsOn",
    "TaxiIn", "CRSArrTime", "ArrTime", "ArrDelay", "ArrDelayMinutes",
    "ArrDel15", "ArrivalDelayGroups", "Cancelled", "CancellationCode",
    "Diverted", "CRSElapsedTime", "ActualElapsedTime", "AirTime", "Distance",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
    "LateAircraftDelay",
]

BTS_RENAME = {
    "Year": "year", "Month": "month", "FlightDate": "flight_date",
    "Reporting_Airline": "reporting_airline",
    "DOT_ID_Reporting_Airline": "dot_id_reporting_airline",
    "IATA_CODE_Reporting_Airline": "iata_code_reporting_airline",
    "Tail_Number": "tail_number",
    "Flight_Number_Reporting_Airline": "flight_number_reporting_airline",
    "OriginAirportID": "origin_airport_id", "Origin": "origin",
    "OriginCityName": "origin_city_name", "OriginState": "origin_state",
    "DestAirportID": "dest_airport_id", "Dest": "dest",
    "DestCityName": "dest_city_name", "DestState": "dest_state",
    "CRSDepTime": "crs_dep_time", "DepTime": "dep_time", "DepDelay": "dep_delay",
    "DepDelayMinutes": "dep_delay_minutes", "DepDel15": "dep_del15",
    "DepartureDelayGroups": "departure_delay_groups", "TaxiOut": "taxi_out",
    "WheelsOff": "wheels_off", "WheelsOn": "wheels_on", "TaxiIn": "taxi_in",
    "CRSArrTime": "crs_arr_time", "ArrTime": "arr_time", "ArrDelay": "arr_delay",
    "ArrDelayMinutes": "arr_delay_minutes", "ArrDel15": "arr_del15",
    "ArrivalDelayGroups": "arrival_delay_groups", "Cancelled": "cancelled",
    "CancellationCode": "cancellation_code", "Diverted": "diverted",
    "CRSElapsedTime": "crs_elapsed_time", "ActualElapsedTime": "actual_elapsed_time",
    "AirTime": "air_time", "Distance": "distance", "CarrierDelay": "carrier_delay",
    "WeatherDelay": "weather_delay", "NASDelay": "nas_delay",
    "SecurityDelay": "security_delay", "LateAircraftDelay": "late_aircraft_delay",
}

@dataclass(frozen=True)
class CacheCheck:
    source: str
    valid: bool
    path: str
    details: str

class FlightDataPull:
    def __init__(
        self,
        project_root: str | Path,
        airport: str = "SFO",
        start_date: str = "2022-01-01",
        end_date_exclusive: str = "2026-06-01",
        noaa_station_id: str = "USW00023234",
        tracked_airlines: tuple[str, ...] = ("AS", "AA", "DL", "F9", "HA", "B6", "WN", "UA"),
    ):
        self.root = Path(project_root).resolve()
        self.airport = airport
        self.start_date = pd.Timestamp(start_date)
        self.end_date_exclusive = pd.Timestamp(end_date_exclusive)
        self.noaa_station_id = str(noaa_station_id)
        self.tracked_airlines = tuple(tracked_airlines)
        self.raw = self.root / "data" / "raw"
        self.bts_path = self.raw / "bts" / "bts_reporting_carrier_2022-01_to_2026-05.parquet"
        self.noaa_root = self.raw / "noaa"
        self.faa_zip = self.raw / "faa" / "ReleasableAircraft.zip"
        self.faa_registry = self.raw / "faa" / "faa_registry_with_deregistered.parquet"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "mids-datasci207-flight-delay-v2/1.0"})

    @property
    def _noaa_years(self) -> range:
        final_date = self.end_date_exclusive - pd.Timedelta(days=1)
        return range(self.start_date.year, final_date.year + 1)

    def _expected_noaa_files(self) -> list[Path]:
        """One GHCNh parquet file per year for the SFO weather station."""
        return [
            self.noaa_root / str(year) / f"GHCNh_{self.noaa_station_id}_{year}.parquet"
            for year in self._noaa_years
        ]

    def _download(self, url: str, destination: Path) -> Path:
        """Download once, using a temporary file so partial data is never cached."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 0:
            return destination
        temporary = destination.with_suffix(destination.suffix + ".part")
        # Reuse the session so sources such as FAA retain landing-page cookies.
        with self.session.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        temporary.replace(destination)
        return destination

    def _bts_check(self, deep: bool = False) -> CacheCheck:
        if not self.bts_path.exists() or self.bts_path.stat().st_size == 0:
            return CacheCheck("BTS", False, str(self.bts_path), "missing or empty")
        try:
            rows = pq.ParquetFile(self.bts_path).metadata.num_rows
            valid = rows >= 1_000_000
            details = f"{rows:,} rows"
            if deep and valid:
                frame = pd.read_parquet(
                    self.bts_path,
                    columns=["flight_date", "origin", "dest", "reporting_airline"],
                )
                dates = pd.to_datetime(frame["flight_date"])
                valid = bool(
                    dates.min() == self.start_date
                    and dates.max() == self.end_date_exclusive - pd.Timedelta(days=1)
                    and ((frame["origin"] == self.airport) | (frame["dest"] == self.airport)).all()
                    and set(self.tracked_airlines).issubset(set(frame["reporting_airline"].dropna()))
                )
                details = f"{len(frame):,} rows; {dates.min().date()} to {dates.max().date()}"
            return CacheCheck("BTS", valid, str(self.bts_path), details)
        except Exception as exc:
            return CacheCheck("BTS", False, str(self.bts_path), f"unreadable: {exc}")

    def _noaa_check(self, deep: bool = False) -> CacheCheck:
        expected = self._expected_noaa_files()
        missing = [path for path in expected if not path.exists() or path.stat().st_size == 0]
        if missing:
            return CacheCheck(
                "NOAA", False, str(self.noaa_root),
                f"{len(missing)} SFO station-years missing",
            )
        if deep:
            invalid = []
            start_utc = self.start_date.tz_localize("UTC")
            end_utc = self.end_date_exclusive.tz_localize("UTC")
            for path in expected:
                try:
                    if pq.ParquetFile(path).metadata.num_rows == 0:
                        invalid.append(path)
                        continue
                    dates = pd.to_datetime(pd.read_parquet(path, columns=["DATE"])["DATE"], utc=True)
                    if dates.min() < start_utc or dates.max() >= end_utc:
                        invalid.append(path)
                except Exception:
                    invalid.append(path)
            if invalid:
                return CacheCheck("NOAA", False, str(self.noaa_root), f"{len(invalid)} invalid station-years")
        return CacheCheck(
            "NOAA", True, str(self.noaa_root),
            f"SFO station {self.noaa_station_id}; {len(expected)} station-years",
        )

    def _faa_check(self, deep: bool = False) -> CacheCheck:
        if not self.faa_zip.exists() or not self.faa_registry.exists():
            return CacheCheck("FAA", False, str(self.faa_registry), "archive or parsed registry missing")
        try:
            rows = pq.ParquetFile(self.faa_registry).metadata.num_rows
            valid = rows >= 500_000 and self.faa_zip.stat().st_size > 50 * 1024 * 1024
            if deep and valid:
                tail = pd.read_parquet(self.faa_registry, columns=["tail_number_clean"])
                valid = bool(tail["tail_number_clean"].notna().mean() > 0.99)
            return CacheCheck("FAA", valid, str(self.faa_registry), f"{rows:,} registry rows")
        except Exception as exc:
            return CacheCheck("FAA", False, str(self.faa_registry), f"unreadable: {exc}")

    def verify(self, deep: bool = False) -> pd.DataFrame:
        """Return a compact status table for all three raw sources."""
        return pd.DataFrame(
            [check.__dict__ for check in (
                self._bts_check(deep), self._noaa_check(deep), self._faa_check(deep)
            )]
        )

    def ensure_all(self, allow_download: bool = True, deep: bool = True) -> pd.DataFrame:
        """Reuse valid files, pull missing sources, then validate the completed cache."""
        checks = self.verify(deep=False).set_index("source")
        for source, pull in (("BTS", self.pull_bts), ("NOAA", self.pull_noaa), ("FAA", self.pull_faa)):
            if not bool(checks.loc[source, "valid"]):
                if not allow_download:
                    raise FileNotFoundError(str(checks.loc[source, "details"]))
                pull()
        result = self.verify(deep=deep)
        if not result["valid"].all():
            raise ValueError("Raw data validation failed:\n" + result.to_string(index=False))
        return result

    def pull_bts(self) -> Path:
        """Download monthly BTS files and cache every flight that touches SFO."""
        pieces: list[pd.DataFrame] = []
        current = self.start_date.to_period("M")
        final = (self.end_date_exclusive - pd.Timedelta(days=1)).to_period("M")
        while current <= final:
            filename = f"{BTS_PREFIX}_{current.year}_{current.month}.zip"
            archive_path = self._download(
                f"{BTS_BASE_URL}/{filename}", self.raw / "bts" / "downloads" / filename
            )
            with zipfile.ZipFile(archive_path) as archive:
                csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
                with archive.open(csv_name) as handle:
                    for chunk in pd.read_csv(
                        handle, usecols=BTS_EXPORT_COLUMNS, chunksize=200_000, low_memory=False
                    ):
                        # Keep all carriers here; the eight-airline cohort is selected downstream.
                        keep = chunk["Origin"].eq(self.airport) | chunk["Dest"].eq(self.airport)
                        if keep.any():
                            pieces.append(chunk.loc[keep].rename(columns=BTS_RENAME))
            current += 1
        frame = pd.concat(pieces, ignore_index=True)
        frame["flight_date"] = pd.to_datetime(frame["flight_date"])
        frame = frame.loc[
            frame["flight_date"].ge(self.start_date)
            & frame["flight_date"].lt(self.end_date_exclusive)
        ].copy()
        frame["source_month"] = frame["flight_date"].dt.strftime("%Y-%m")
        frame["tail_number_clean"] = frame["tail_number"].astype("string").str.strip().str.upper()
        self.bts_path.parent.mkdir(parents=True, exist_ok=True)
        frame.sort_values(
            ["flight_date", "origin", "dest", "reporting_airline", "flight_number_reporting_airline"]
        ).to_parquet(self.bts_path, index=False)
        return self.bts_path

    def pull_noaa(self) -> list[Path]:
        """Download SFO station files and clip them to the configured date window."""
        outputs = []
        start_utc = self.start_date.tz_localize("UTC")
        end_utc = self.end_date_exclusive.tz_localize("UTC")
        for year in self._noaa_years:
            name = f"GHCNh_{self.noaa_station_id}_{year}.parquet"
            url = f"{NOAA_BASE_URL}/access/by-year/{year}/parquet/{name}"
            path = self._download(url, self.noaa_root / str(year) / name)
            frame = pd.read_parquet(path)
            observed = pd.to_datetime(frame["DATE"], errors="coerce", utc=True)
            frame = frame.loc[observed.ge(start_utc) & observed.lt(end_utc)].copy()
            frame.to_parquet(path, index=False)
            outputs.append(path)
        return outputs

    @staticmethod
    def _read_faa_table(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
        member = next(item for item in archive.namelist() if item.upper().endswith(name.upper()))
        with archive.open(member) as handle:
            frame = pd.read_csv(handle, dtype="string", encoding="utf-8-sig", low_memory=False)
        frame.columns = [str(column).strip() for column in frame.columns]
        return frame

    @staticmethod
    def _read_faa_deregistered(archive: zipfile.ZipFile) -> pd.DataFrame:
        member = next(item for item in archive.namelist() if item.upper().endswith("DEREG.TXT"))
        rows = []
        with archive.open(member) as handle:
            next(handle, None)
            for raw_line in handle:
                parts = raw_line.decode("utf-8-sig", errors="replace").rstrip("\r\n").split(",")
                if len(parts) >= 23:
                    rows.append({
                        "N-NUMBER": parts[0], "MFR MDL CODE": parts[2],
                        "ENG MFR MDL": parts[10], "YEAR MFR": parts[11],
                        "STATUS CODE": parts[3], "AIR WORTH DATE": parts[16],
                        "CANCEL DATE": parts[17], "LAST ACTION DATE": parts[21],
                        "CERT ISSUE DATE": parts[22], "FAA_RECORD_TYPE": "DEREG",
                    })
        return pd.DataFrame(rows)

    def _parse_faa(self) -> pd.DataFrame:
        """Combine active and deregistered FAA records into one row per tail."""
        with zipfile.ZipFile(self.faa_zip) as archive:
            master = self._read_faa_table(archive, "MASTER.txt")
            aircraft = self._read_faa_table(archive, "ACFTREF.txt")
            engines = self._read_faa_table(archive, "ENGINE.txt")
            dereg = self._read_faa_deregistered(archive)
        aircraft = aircraft.rename(columns={
            "CODE": "MFR MDL CODE", "TYPE-ACFT": "TYPE AIRCRAFT REF",
            "TYPE-ENG": "TYPE ENGINE REF",
        })
        engines = engines.rename(columns={"CODE": "ENG MFR MDL", "MFR": "MFR_ENG", "MODEL": "MODEL_ENG"})
        active = master.merge(aircraft, on="MFR MDL CODE", how="left", suffixes=("", "_REF"))
        active = active.merge(engines, on="ENG MFR MDL", how="left")
        active["FAA_RECORD_TYPE"] = "MASTER"
        dereg = dereg.merge(aircraft, on="MFR MDL CODE", how="left", suffixes=("", "_REF"))
        dereg = dereg.merge(engines, on="ENG MFR MDL", how="left")
        combined = pd.concat([active, dereg], ignore_index=True, sort=False)
        columns = {
            "N-NUMBER": "n_number", "YEAR MFR": "year_mfr",
            "TYPE AIRCRAFT": "type_aircraft", "TYPE AIRCRAFT REF": "type_aircraft_ref",
            "TYPE ENGINE": "type_engine", "TYPE ENGINE REF": "type_engine_ref",
            "MFR MDL CODE": "mfr_mdl_code", "ENG MFR MDL": "eng_mfr_mdl",
            "MFR": "mfr", "MODEL": "model", "NO-ENG": "no_eng",
            "NO-SEATS": "no_seats", "AC-WEIGHT": "ac_weight", "SPEED": "speed",
            "MFR_ENG": "mfr_eng", "MODEL_ENG": "model_eng",
            "HORSEPOWER": "horsepower", "THRUST": "thrust",
            "STATUS CODE": "status_code", "AIR WORTH DATE": "air_worth_date",
            "CANCEL DATE": "cancel_date", "LAST ACTION DATE": "last_action_date",
            "CERT ISSUE DATE": "cert_issue_date", "EXPIRATION DATE": "expiration_date",
            "MODE S CODE": "mode_s_code", "UNIQUE ID": "unique_id",
            "FAA_RECORD_TYPE": "faa_record_type",
        }
        combined = combined.rename(columns=columns)
        combined["type_aircraft"] = combined.get("type_aircraft").fillna(combined.get("type_aircraft_ref"))
        combined["type_engine"] = combined.get("type_engine").fillna(combined.get("type_engine_ref"))
        keep = [value for value in columns.values() if value in combined.columns]
        combined = combined[keep].copy()
        combined["tail_number_clean"] = "N" + combined["n_number"].astype("string").str.strip().str.upper()
        combined["faa_match"] = True
        priority = combined["faa_record_type"].map({"MASTER": 0, "DEREG": 1}).fillna(2)
        return (
            combined.assign(_priority=priority)
            .sort_values(["tail_number_clean", "_priority"], kind="stable")
            .drop_duplicates("tail_number_clean", keep="first")
            .drop(columns="_priority")
            .reset_index(drop=True)
        )

    def pull_faa(self) -> tuple[Path, Path]:
        """Download the current FAA archive and cache the parsed tail registry."""
        if not self.faa_zip.exists() or self.faa_zip.stat().st_size == 0:
            page = self.session.get(FAA_DOWNLOAD_PAGE, timeout=(30, 120))
            page.raise_for_status()
            soup = BeautifulSoup(page.text, "html.parser")
            archive_url = next(
                (urljoin(page.url, anchor["href"]) for anchor in soup.find_all("a", href=True)
                 if "Download the Aircraft Registration Database" in " ".join(anchor.get_text(" ", strip=True).split())),
                None,
            )
            if archive_url is None:
                raise RuntimeError("Could not discover the FAA registry archive URL")
            self._download(archive_url, self.faa_zip)
        if not self.faa_registry.exists() or self.faa_registry.stat().st_size == 0:
            self._parse_faa().to_parquet(self.faa_registry, index=False)
        return self.faa_zip, self.faa_registry