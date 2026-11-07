---
title: Smallprint
emoji: 🔎
colorFrom: gray
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Flags potentially unfair terms in ToS and privacy policies
models:
  - mukeshkbj/smallprint-tiny-aya-water-3b
---

# Smallprint

Paste a Terms of Service or privacy policy. Smallprint splits it into clauses, runs a fine-tuned
Tiny Aya Water (3.35B) model on each one, and lists the clauses that may be unfair to you, each with
an exact quote and a plain-language explanation.

Code, data pipeline and evaluation: https://github.com/mukeshkbj/smallprint

Not legal advice.
