"""Bake a pre-rendered first screen into Streamlit's index.html.

Streamlit renders everything in the browser, so its index.html is an empty
shell titled "Streamlit". This adds real meta tags (search results, link
previews) and a static copy of the landing screen that shows instantly and is
removed once the live app has drawn its hero. Run at Docker build time; it
fails loudly if a Streamlit upgrade changes the file it edits.
"""
import os
import pathlib
import sys

import streamlit

HERE = pathlib.Path(__file__).parent
SITE_URL = os.environ.get("SITE_URL", "https://chatpdf.siddharthranjan.app").rstrip("/")
MARKER = "<!-- chatpdf-prerender -->"

index = pathlib.Path(streamlit.__file__).parent / "static" / "index.html"
html = index.read_text(encoding="utf-8")
if MARKER in html:
    sys.exit(f"{index} is already patched")

head = (HERE / "head.html").read_text(encoding="utf-8").replace("{{SITE_URL}}", SITE_URL)
body = (HERE / "body.html").read_text(encoding="utf-8")

replacements = [
    ("<title>Streamlit</title>", MARKER + head),
    ('<link rel="shortcut icon" href="./favicon.png" />', ""),
    ("<noscript>You need to enable JavaScript to run this app.</noscript>", ""),
    ('<div id="root"></div>', body),
]
for old, new in replacements:
    if html.count(old) != 1:
        sys.exit(f"prerender: expected exactly one {old!r} in {index}; Streamlit's index.html changed")
    html = html.replace(old, new)

index.write_text(html, encoding="utf-8")
print(f"prerender: patched {index}")
