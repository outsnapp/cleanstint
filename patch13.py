with open('app/dashboard.py', 'r') as f:
    t = f.read()
ok = lambda n, c: print(("✅ " if c else "❌ MISS ") + n)

# 1. Kill the empty 100vh shell div (open)
old = "st.markdown('<div class=\"cleanstint-shell\">', unsafe_allow_html=True)"
ok("shell open removed", old in t); t = t.replace(old, "# shell wrapper removed: Streamlit fragments auto-close divs")

# 2. Kill the shell close
old = "st.markdown('</div>', unsafe_allow_html=True)"
ok("shell close removed", old in t); t = t.replace(old, "")

# 3. Add mockup hairlines: divider above footer + chart/stats vertical border
old = "</style>"
new = """.stats { border-left: 1px solid #22272c; }
[data-testid="stPlotlyChart"] { border-right: 1px solid #22272c; padding-right: 12px; }
</style>"""
ok("css hairlines added", old in t); t = t.replace(old, new, 1)

old = 'st.markdown(f"""\n<div class="footer">'
new = 'st.markdown(\'<div class="topbar-rule"></div>\', unsafe_allow_html=True)\nst.markdown(f"""\n<div class="footer">'
ok("footer top rule added", old in t); t = t.replace(old, new, 1)

with open('app/dashboard.py', 'w') as f:
    f.write(t)
print("🎉 patch13 complete")
