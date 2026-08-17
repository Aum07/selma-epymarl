"""SELMA Stage D: monetary cost, recomputed from tokens and a dated price table.

**No price is hard-coded here.** Rates arrive as configuration together with the
date they were checked, and a run that wants a cost ceiling must supply both.
That is deliberate and long-standing in this codebase: ``ProviderUsage.cost_usd``
is always ``None`` and the guidance log writes ``n/a``, because a number baked
into source silently becomes stale evidence the moment a vendor changes a price.
A cost in a report is only meaningful next to the date its table was read.

The units are USD per **million** tokens, matching how vendors publish them.

Cached input is priced separately and is **not** the application's SQLite
response cache. They are unrelated: a provider-side cached-input token is a
prompt prefix the vendor recognised and billed at a lower rate on a request that
really happened; a SQLite cache hit means no request happened at all and costs
nothing. Conflating them would double-count savings.
"""

from typing import NamedTuple, Optional


class PriceTableError(ValueError):
    """A cost ceiling was requested without a usable dated price table."""


class PriceTable(NamedTuple):
    """Dated per-million-token rates. ``date`` is part of the evidence."""

    date: str
    input_usd_per_mtok: float
    cached_input_usd_per_mtok: float
    output_usd_per_mtok: float

    @property
    def usable(self):
        """True when a cost really can be computed from this table."""
        return bool(self.date) and any(
            r > 0.0 for r in (self.input_usd_per_mtok,
                              self.cached_input_usd_per_mtok,
                              self.output_usd_per_mtok))

    def cost_usd(self, *, input_tokens, cached_input_tokens, output_tokens):
        """Cost of a token bundle under this table.

        ``input_tokens`` is the provider's total input count, which **includes**
        the cached ones; the non-cached remainder is what is billed at the full
        input rate. Getting that wrong overstates cost, so it is done here once
        rather than at each call site.
        """
        total_input = int(input_tokens or 0)
        cached = min(int(cached_input_tokens or 0), total_input)
        non_cached = total_input - cached
        return (non_cached * self.input_usd_per_mtok / 1_000_000.0
                + cached * self.cached_input_usd_per_mtok / 1_000_000.0
                + int(output_tokens or 0) * self.output_usd_per_mtok / 1_000_000.0)


#: A table with no rates and no date: cost is *not calculated*, and any attempt
#: to enforce a ceiling with it is a configuration error rather than a silent
#: pass. This is the default everywhere.
NO_PRICE_TABLE = PriceTable("", 0.0, 0.0, 0.0)


def price_table_from_args(args, getter):
    """Build a :class:`PriceTable` from the SELMA config surface."""
    return PriceTable(
        date=str(getter(args, "selma_price_table_date", "") or ""),
        input_usd_per_mtok=float(
            getter(args, "selma_price_input_usd_per_mtok", 0.0) or 0.0),
        cached_input_usd_per_mtok=float(
            getter(args, "selma_price_cached_input_usd_per_mtok", 0.0) or 0.0),
        output_usd_per_mtok=float(
            getter(args, "selma_price_output_usd_per_mtok", 0.0) or 0.0),
    )


class CostBudget:
    """A conservative spend ceiling checked **before** every live attempt.

    The projection is deliberately pessimistic: the next attempt is assumed to
    cost the largest input it has ever seen (never fewer than
    ``floor_input_tokens``) plus the full ``max_output_tokens`` allowance, all at
    the uncached rate. A run therefore stops short of the ceiling rather than
    discovering it in the bill.

    ``cap_usd = None`` disables the ceiling and is the default; a ceiling without
    a usable dated table raises rather than silently permitting everything.
    """

    def __init__(self, *, cap_usd=None, price_table=NO_PRICE_TABLE,
                 max_output_tokens=512, floor_input_tokens=4000):
        if cap_usd is not None and not price_table.usable:
            raise PriceTableError(
                "a cost ceiling of ${:.4f} was requested without a dated price "
                "table. Supply selma_price_table_date and the per-million-token "
                "rates; no price is hard-coded in this build."
                .format(float(cap_usd))
            )
        self.cap_usd = None if cap_usd is None else float(cap_usd)
        self.price_table = price_table
        self.max_output_tokens = int(max_output_tokens)
        self.floor_input_tokens = int(floor_input_tokens)
        self.observed_max_input_tokens = 0

    def note_usage(self, input_tokens):
        if input_tokens:
            self.observed_max_input_tokens = max(
                self.observed_max_input_tokens, int(input_tokens))

    def worst_case_next_attempt_usd(self):
        assumed_input = max(self.observed_max_input_tokens, self.floor_input_tokens)
        return self.price_table.cost_usd(
            input_tokens=assumed_input, cached_input_tokens=0,
            output_tokens=self.max_output_tokens)

    def would_exceed(self, spent_usd):
        """True when the *next* attempt could push the run past the ceiling."""
        if self.cap_usd is None:
            return False
        return (float(spent_usd) + self.worst_case_next_attempt_usd()) > self.cap_usd


def accounting_cost_usd(accounting, price_table) -> Optional[float]:
    """Cost of everything an accounting ledger has recorded, or ``None``.

    ``None`` means *not calculated* -- no dated table was supplied -- and must
    never be rendered as ``0.0``.
    """
    if not price_table.usable:
        return None
    return price_table.cost_usd(
        input_tokens=accounting.input_tokens,
        cached_input_tokens=accounting.cached_input_tokens,
        output_tokens=accounting.output_tokens,
    )
