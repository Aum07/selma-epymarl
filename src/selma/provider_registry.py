"""SELMA provider registry.

Kept separate from :mod:`selma.provider` on purpose. That module is the frozen
Stage C schema/validator/mock and is asserted by
``tests/test_selma_stage_c_pbrs.py`` to contain no network, credential or
vendor identifiers at all. Naming a vendor there would break that guarantee, so
the vendor-aware registry lives here and the adapter is imported lazily, only
when that provider is actually requested.
"""

PROVIDER_NONE = "none"
PROVIDER_MOCK = "deterministic_mock"
PROVIDER_OPENAI = "openai"

#: Providers this build can construct. ``none`` is not here: it means "no
#: provider", and active mode must name one explicitly.
SUPPORTED_PROVIDERS = (PROVIDER_MOCK, PROVIDER_OPENAI)

#: Providers that can perform network I/O. Everything else is offline by
#: construction, which is what keeps off/shadow/mock runs free of any SDK.
LIVE_PROVIDERS = (PROVIDER_OPENAI,)


def is_live_provider(name):
    return str(name) in LIVE_PROVIDERS


def make_provider(name, **kwargs):
    """Construct a provider by name. Unsupported names fail loudly.

    The deterministic mock is constructed through :mod:`selma.provider`, so the
    Stage C path is unchanged. The OpenAI adapter is imported only here, only
    when named.
    """
    name = str(name)
    if name == PROVIDER_MOCK:
        from selma.provider import DeterministicMockProvider

        return DeterministicMockProvider()
    if name == PROVIDER_OPENAI:
        from selma.openai_provider import OpenAIGuidanceProvider

        return OpenAIGuidanceProvider(**kwargs)
    raise ValueError(
        "selma_provider={!r} is not supported in this build. Available: {}."
        .format(name, ", ".join(sorted(SUPPORTED_PROVIDERS)))
    )
