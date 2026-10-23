from newsagg import enrich, filters

PAGE = """<html><head>
<meta property="og:description" content="Seven people were killed when gunmen opened fire.">
<meta property="og:image" content="https://example.com/church.jpg">
<script type="application/ld+json">{"@context": "https://schema.org", "@graph": [
  {"@type": "WebSite", "name": "Example"},
  {"@type": ["NewsArticle"], "articleSection": ["World", "Middle East"],
   "author": [{"@type": "Person", "name": "A. Reporter"}]}
]}</script>
<script type="application/ld+json">{ not valid json </script>
</head><body>Body text we never keep.</body></html>"""


def test_parse_page_reads_metadata():
    info = enrich.parse_page(PAGE)
    assert info.description.startswith("Seven people")
    assert info.image == "https://example.com/church.jpg"
    assert info.ld_types == ["NewsArticle"]          # found inside @graph
    assert info.section == "World, Middle East"
    assert info.has_byline                            # from JSON-LD author


def test_parse_page_handles_bare_page():
    info = enrich.parse_page("<html><head><title>x</title></head></html>")
    assert info == enrich.PageInfo()


def test_page_type_rule():
    assert filters.page_type(enrich.PageInfo(ld_types=["OpinionNewsArticle"])) == "page_type"
    assert filters.page_type(enrich.PageInfo(section="Opinion")) == "page_type"
    assert filters.page_type(enrich.PageInfo(ld_types=["NewsArticle"], section="World")) is None
