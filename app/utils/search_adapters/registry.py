from types import MappingProxyType

from app.utils.search_adapters.arxiv import ArxivAdapter
from app.utils.search_adapters.contracts import SearchAdapter
from app.utils.search_adapters.dblp import DblpAdapter
from app.utils.search_adapters.google_scholar import GoogleScholarAdapter

_ADAPTERS = MappingProxyType(
    {
        adapter.provider_id: adapter
        for adapter in (ArxivAdapter(), DblpAdapter(), GoogleScholarAdapter())
    }
)


def get_adapter(provider_id: str) -> SearchAdapter:
    try:
        return _ADAPTERS[provider_id]
    except KeyError:
        raise ValueError(f"Unsupported search provider: {provider_id}") from None


def supported_provider_ids() -> tuple[str, ...]:
    return tuple(_ADAPTERS)
