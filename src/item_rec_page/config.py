from dataclasses import dataclass
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    riot_api_key: str | None
    platform: str
    region: str
    data_dir: Path
    model_dir: Path
    item_catalog_path: Path
    liveclient_base_url: str
    verify_liveclient_ssl: bool

    @classmethod
    def from_env(
        cls,
        riot_api_key: str | None = None,
        platform: str | None = None,
        region: str | None = None,
    ) -> "Settings":
        data_dir = PROJECT_ROOT / "data"
        model_dir = PROJECT_ROOT / "models"
        return cls(
            riot_api_key=riot_api_key or os.getenv("RIOT_API_KEY"),
            platform=(platform or os.getenv("RIOT_PLATFORM", "NA1")).upper(),
            region=(region or os.getenv("RIOT_REGION", "AMERICAS")).upper(),
            data_dir=data_dir,
            model_dir=model_dir,
            item_catalog_path=data_dir / "item_catalog.json",
            liveclient_base_url=os.getenv("LIVECLIENT_BASE_URL", "https://127.0.0.1:2999/liveclientdata"),
            verify_liveclient_ssl=os.getenv("LIVECLIENT_VERIFY_SSL", "false").lower() == "true",
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
