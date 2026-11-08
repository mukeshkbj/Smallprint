# Abstract (draft, under 500 words)

> Numbers are base -> SFT (the published model). Details and both DPO runs: docs/results.md.

**Smallprint: a 3B on-device model that reads the small print**

Almost nobody reads Terms of Service or privacy policies, yet they routinely let providers change the
terms at will, close accounts without notice, remove content, limit their liability and send disputes to
arbitration. Tools that summarise these documents usually send the full text to a cloud API. That is
costly at scale, and awkward for a privacy tool.

Smallprint is Tiny Aya Water (3.35B), fine-tuned to read a policy one clause at a time. For each clause it
returns JSON: whether the clause contains a potentially unfair term, which of nine categories apply, an
exact quote from the clause as evidence, and one plain sentence in the clause's language. A document
report is assembled from the clause results with no extra model call, so it cannot state anything a quoted
clause does not support. The model works in English, German, Italian and Polish and is small enough to run
on a laptop.

Data comes from expert annotations: LexGLUE UNFAIR-ToS (English) and the multilingual CLAUDETTE corpus.
Because both derive from the same project, we removed train/test leakage at company level (including
translations) and at sentence level before training. A teacher model wrote the evidence and explanations for
unfair clauses only; it never changed an expert label. Every quote was checked to be verbatim, and CuratorKIT's
grounding gate checked every explanation against its clause. We trained with AlignTune (LoRA SFT), then tried DPO on the SFT model's own mistakes plus targeted
negatives. The first DPO run tripled false alarms because its pairs favoured flagging two to one; a
label-balanced rerun came close but still scored below SFT. AuditKIT caught both before release, and the
published model is the one that scored best.

We evaluated with AuditKIT on the LexGLUE test set and on held-out CLAUDETTE companies in four languages,
using the same prompt before and after. On LexGLUE, valid-JSON output rose from 25% to 99.9%, unfair-term F1
from 0.26 to 0.79, and category macro-F1 from 0.15 to 0.76. False alarms on fair clauses fell from 23% to 4.5%,
and the share of evidence that is a real quote rose from 74% to 99%. In German, Italian and Polish,
category macro-F1 rose from 0.15–0.17 to 0.50–0.61, with Polish the weakest.

At [N] documents a month, the closed-model API we compared against would cost about $[X], self-hosting Smallprint
on a [GPU] about $[X], and running it on the user's own device nothing, with no document leaving the device.

Limitations: labels cover nine categories from EU consumer-law research and sentence-level units. The model can
miss unfair terms that span several sentences. The base model's licence is non-commercial. Smallprint points at
clauses worth reading; it is not legal advice.
