"""The tokenization variants of eval/tokenization.py on hand-picked text (no model tokenizer needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))   # the pool's fresh processes import it by name
import tokenization  # noqa: E402


def test_words_keep_their_vowel_signs():
    # Thai, Hindi and Telugu vowel signs are combining marks; `\w+` cut the words at them
    assert tokenization.words_marks("การเรียนรู้ของเครื่อง") == ["การเรียนรู้ของเครื่อง"]
    assert tokenization.words_marks("मशीन लर्निंग क्या है") == ["मशीन", "लर्निंग", "क्या", "है"]
    assert tokenization.words_marks("మెషిన్ లెర్నింగ్") == ["మెషిన్", "లెర్నింగ్"]


def test_spaced_scripts_are_cut_as_today():
    from peresearch.zetokrag import core
    for text in ["Học máy là gì? SuperBPE, Qwen3.6-35B-A3B-FP8", "Maschinelles Lernen", "машинное обучение",
                 "تعلم الآلة", "hoc may la gi"]:
        assert tokenization.words_marks(text) == core.tokenize(text, folded=False), text
        assert tokenization.split_unspaced(text, tokenization.bigrams) == core.tokenize(text, folded=False), text


def test_unspaced_scripts_become_character_pairs_and_other_words_stay_whole():
    cut = lambda t: tokenization.split_unspaced(t, tokenization.bigrams)
    assert cut("机器学习") == ["机器", "器学", "学习"]
    assert cut("GPT-4を使う") == ["gpt", "4", "を使", "使う"]
    assert cut("コンピューター") == ["コン", "ンピ", "ピュ", "ュー", "ータ", "ター"]      # ー is shared by the kana
    assert cut("的") == ["的"]
    assert cut("เรียนรู้") == ["เรี", "รีย", "ยน", "นรู้"]                     # grapheme clusters keep their marks


def test_reciprocal_rank_fusion_merges_two_rankings():
    fused = tokenization.rrf([[1, 2, 3]], [[3, 1, 9]])[0]
    assert fused[:2] == [1, 3] and set(fused) == {1, 2, 3, 9}      # in both lists beats in one; 1 is ranked higher


def test_cutting_on_all_cores_gives_the_terms_of_cutting_one_text_at_a_time():
    texts = ["Học máy là gì? SuperBPE", "机器学习模型的训练 and GPT-4を使う", "การเรียนรู้ของเครื่อง",
             "मशीन लर्निंग क्या है", "hoc may la gi", ""] * 400                     # 2,400 texts: the process pool
    for variant, one in tokenization.REGEX_CUTS.items():
        assert tokenization.cut_all(variant, texts) == [one(t) for t in texts], variant
