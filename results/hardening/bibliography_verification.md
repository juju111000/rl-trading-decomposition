# Bibliographic verification (hardening session, Part 3)

Verified directly against primary sources (arXiv abstract pages + ar5iv HTML
renderings of the full text) on 2026-08-06. **Before checking specific
claims: no "Table 9" (or any manuscript/paper draft, `.tex`, `.bib`, or
similarly-named file) exists anywhere in this repository** (`grep`/`glob`
search came back empty). I cannot check our Table 9 against these sources
because I do not have access to Table 9's actual content from within this
codebase — if it lives in a separate manuscript repo or document, point me
to it and I'll re-run this check against it. Everything below is the
verified ground truth from the primary sources themselves, ready to check
against Table 9 once it's available.

## Jiang, Xu & Liang 2017 -- arXiv:1706.10059

**Title (confirmed):** "A Deep Reinforcement Learning Framework for the
Financial Portfolio Management Problem"

This is a **cryptocurrency** portfolio-management paper (Poloniex
exchange), not equities -- worth noting if Table 9 cites it as an
equities comparison.

- **Data frequency:** 30-minute bars. Quote (Section 2.1): *"T = 30
  minutes in all experiments of this paper."*
- **Commission rate:** 0.25% per trade, both buy and sell. Quote (Section
  2.3): *"a single constant commission rate for both selling and
  purchasing for all non-cash assets is used, c_s = c_p = 0.25%, the
  maximum rate at Poloniex."*
- **Backtest window:** **NOT a single window** -- three separate ~52-day
  back-tests, not one continuous test period:
  - Back-test 1: 2016-09-07 to 2016-10-28 (52 days)
  - Back-test 2: 2016-12-08 to 2017-01-28 (52 days)
  - Back-test 3: 2017-03-07 to 2017-04-27 (52 days)
  (Table 6.1; abstract's "50 days" is a rounded figure for the same three
  windows.)

**If Table 9 states a single backtest length (e.g. "50 days") without
noting there were three separate non-contiguous windows, or states a
daily data frequency, or an equities test, those need correction.**

## "Yang et al. 2020" -- arXiv ID mismatch found

**The arXiv ID given, arXiv:2011.09607, is NOT a Yang-first-authored
paper.** Verified directly from the abstract page: it is **Liu, Yang,
Chen, Zhang, Yang, Xiao & Wang (2020), "FinRL: A Deep Reinforcement
Learning Library for Automated Stock Trading in Quantitative Finance"**
-- first author Xiao-Yang Liu; Hongyang Yang is second author. Submitted
2020-11-19.

The paper actually titled **"Deep Reinforcement Learning for Automated
Stock Trading: An Ensemble Strategy"** (Hongyang Yang, Xiao-Yang Liu,
Shan Zhong, Anwar Walid -- genuinely Yang-first-authored, ACM ICAIF 2020)
carries a **different arXiv ID: 2511.12120** (a late arXiv upload,
submitted 2025-11-15, of the original 2020 ICAIF paper -- the arXiv
number's year prefix reflects the upload date, not the original
publication date).

**Table 9 needs one of two corrections, depending on which paper was
actually intended:**
- If Table 9 means the ensemble-strategy paper (PPO+A2C+DDPG ensemble) --
  the citation should read **arXiv:2511.12120**, not 2011.09607, and the
  author should not be shortened to "Yang et al." if it's actually the
  FinRL library paper being cited.
- If Table 9 means the FinRL *library* paper -- the citation is
  correctly arXiv:2011.09607, but the author-year should read **"Liu et
  al. 2020,"** not "Yang et al. 2020" (Yang is second author).

Verified details for **both** candidates, since I don't know which one
Table 9 intends:

### If the intended citation is Yang, Liu, Zhong & Walid, "...An Ensemble
Strategy" (arXiv:2511.12120)
- **Transaction cost:** 0.1% of trade value, each way. Quote (Section
  III-B, eq. 2): *"we assume our transaction costs to be 0.1% of the
  value of each trade (either buy or sell)."*
- **Number of runs/seeds:** **Not reported.** No mention of repeated
  trials or random seeds anywhere in the methodology or experiments
  sections (VI-A, VI-B) -- if Table 9 attributes a specific seed count
  to this paper, that number isn't in the paper and needs to be dropped
  or sourced elsewhere.
- **Evaluation period:** 2016-01-01 to 2020-05-08 (out-of-sample test).
  Quote (Section VI-A): *"we test our agent's performance on trading
  data, which is the unseen out-of-sample data from 01/01/2016 to
  05/08/2020."* Also given as "(2016/01/04-2020/05/08)" in Table II's
  caption (the 4-day discrepancy between section text and table caption
  is in the source paper itself, not introduced here).

### If the intended citation is Liu, Yang, Chen, Zhang, Yang, Xiao & Wang,
"FinRL..." (arXiv:2011.09607)
- **Transaction cost:** the paper only describes the MECHANISM, not a
  value actually used in its own demonstrations. Quote (Section 4.4):
  *"Per share percentage: a per share rate for every share traded, for
  example, 1/1000 or 2/1000 are the most commonly used transaction cost
  rate for each trade."* Tables 1-2 and Figures 4-5 (the actual
  demonstration results) do not state which rate was applied -- **no
  single transaction-cost figure can be attributed to this paper's
  results without over-claiming.**
- **Number of runs/seeds:** **Not reported**, same as above.
- **Evaluation period:** 2019-01-01 to 2020-09-23, per both Table 1
  (single-stock) and Table 2 (multi-stock/portfolio-allocation).

## Action items once Table 9 is available

1. Fix the arXiv ID / author-year pairing per whichever paper was
   actually intended (see above).
2. If Table 9 states a specific transaction-cost figure for the FinRL
   library paper's own results, that number is not traceable to the
   paper itself and should be removed or re-sourced.
3. If Table 9 attributes a seed/run count to either paper, neither paper
   reports one -- that number needs a different source or should be
   dropped.
4. If Table 9 describes Jiang et al.'s backtest as one continuous window,
   correct to "three non-contiguous ~52-day windows."
