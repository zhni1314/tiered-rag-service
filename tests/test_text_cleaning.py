from tiered_rag.text_cleaning import strip_repeating_layout


def test_removes_repeated_page_header_and_footer():
    raw = "某学院 某系 v1.0 版\n第一章 概述\n12 / 117\n正文内容\n"
    out = strip_repeating_layout(
        raw,
        header_pattern=r"某学院\s+某系\s+v1\.\d+\s*版\s*",
        footer_pattern=r"\s*\d+\s*/\s*\d+\s*",
    )
    assert "某学院" not in out
    assert "12 / 117" not in out
    assert "正文内容" in out


def test_joins_cjk_sentences_split_by_soft_wrapping():
    raw = "这是一句被排版\n折断的中文句子。\n\n下一段另起。"
    out = strip_repeating_layout(raw)
    assert "折断" in out.replace("\n", "")
    assert "被排版折断的中文句子。" in out
    assert "\n\n下一段另起。" in out, "paragraph breaks must survive"


def test_keeps_line_break_before_non_cjk_content():
    raw = "中文结尾\nEnglish next"
    assert "\n" in strip_repeating_layout(raw)


def test_cjk_break_joining_can_be_disabled():
    raw = "这是一句被排版\n折断的中文句子。"
    assert "\n" in strip_repeating_layout(raw, drop_cjk_line_breaks=False)


def test_empty_patterns_are_a_no_op():
    out = strip_repeating_layout("原始文本\n")
    assert out == "原始文本\n"
