"""Render results.md to results.pdf with headless Chrome (no extra Python deps).

    .venv/bin/python bench/md2pdf.py results.md results.pdf

Supports the Markdown subset the report uses: #/##/### headings, paragraphs,
pipe tables, bullet and numbered lists, **bold**, *italic*, `code`, fenced
code blocks, horizontal rules and images.
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 10.2pt; line-height: 1.38; color: #1a1a1a; }
h1 { font-size: 20pt; margin: 0 0 4pt 0; }
h2 { font-size: 14pt; margin: 18pt 0 6pt 0; border-bottom: 1.5px solid #333; padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt 0; page-break-after: avoid; }
p { margin: 4pt 0 6pt 0; }
ul, ol { margin: 2pt 0 6pt 0; padding-left: 18pt; }
li { margin: 1.5pt 0; }
table { border-collapse: collapse; margin: 5pt 0 9pt 0; font-size: 9pt; }
tr { page-break-inside: avoid; }
tr:first-child { page-break-after: avoid; }
th, td { border: 1px solid #bbb; padding: 2.5pt 5pt; text-align: left; vertical-align: top; }
th { background: #eee; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.8pt; background: #f3f3f3; padding: 0 2pt; }
pre { background: #f3f3f3; padding: 6pt; font-size: 8.5pt; white-space: pre-wrap; border: 1px solid #ddd; }
hr { border: 0; border-top: 1px solid #999; margin: 10pt 0; }
img { max-width: 100%; }
.small { color: #555; font-size: 9pt; }
blockquote { border-left: 3px solid #999; margin: 4pt 0; padding: 2pt 8pt; color: #333; background: #fafafa; }
"""


def inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![*\w])\*([^*]+)\*(?![*\w])", r"<i>\1</i>", s)
    s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    return s


def convert(md: str, base: Path) -> str:
    lines = md.splitlines()
    out, i = [], 0
    in_list = None
    para: list[str] = []

    def flush_para():
        nonlocal para
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para = []

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(f"</{in_list}>")
            in_list = None

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            flush_para(); close_list()
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                buf.append(lines[j]); j += 1
            out.append("<pre>" + html.escape("\n".join(buf)) + "</pre>")
            i = j + 1
            continue
        if not s:
            flush_para(); close_list(); i += 1; continue
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            flush_para(); close_list()
            out.append(f"<h{len(m.group(1))}>{inline(m.group(2))}</h{len(m.group(1))}>")
            i += 1; continue
        if s in ("---", "***"):
            flush_para(); close_list(); out.append("<hr>"); i += 1; continue
        if s.startswith("|"):
            flush_para(); close_list()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip()); i += 1
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            body = [c for c in cells if not all(re.fullmatch(r":?-{2,}:?", x) for x in c)]
            if not body:
                continue
            t = ["<table>", "<tr>" + "".join(f"<th>{inline(c)}</th>" for c in body[0]) + "</tr>"]
            for r in body[1:]:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</table>")
            out.append("\n".join(t))
            continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", ln)
        if m:
            flush_para()
            kind = "ol" if m.group(2)[0].isdigit() else "ul"
            if in_list != kind:
                close_list()
                start = f' start="{int(m.group(2)[:-1])}"' if kind == "ol" else ""
                out.append(f"<{kind}{start}>"); in_list = kind
            text = m.group(3)
            # continuation lines
            j = i + 1
            while j < len(lines) and lines[j].startswith("  ") and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[j]):
                text += " " + lines[j].strip(); j += 1
            out.append(f"<li>{inline(text)}</li>")
            i = j; continue
        if s.startswith(">"):
            flush_para(); close_list()
            out.append("<blockquote>" + inline(s.lstrip("> ")) + "</blockquote>")
            i += 1; continue
        para.append(s); i += 1
    flush_para(); close_list()
    return "<!doctype html><html><head><meta charset='utf-8'><style>" + CSS + "</style></head><body>" + "\n".join(out) + "</body></html>"


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    htmlp = Path(__file__).resolve().parent / "out" / (dst.stem + ".html")
    htmlp.parent.mkdir(parents=True, exist_ok=True)
    page = convert(src.read_text(), src.parent)
    # image paths in the Markdown are relative to the Markdown file, not to bench/out
    page = re.sub(r'src="(?!/|file:|https?:)([^"]+)"',
                  lambda m: f'src="{(src.parent / m.group(1)).resolve()}"', page)
    htmlp.write_text(page)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                    f"--print-to-pdf={dst.resolve()}", str(htmlp.resolve())], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {dst} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
