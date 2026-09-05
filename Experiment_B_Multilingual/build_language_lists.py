#!/usr/bin/env python
"""Step 1 - build the authoritative pre-training language lists and the FLEURS mapping.

Sources (recorded in the emitted JSON):
  XLSR-53   fairseq examples/wav2vec/README.md - the model's own repo, which enumerates
            the three constituent corpora and their languages explicitly. The XLSR paper
            (Conneau et al. 2020) only names the *experimental subsets* (10 CommonVoice,
            10 BABEL, 8 MLS languages), not the full 53, so the README is the right source.
  XLS-R     Table 1 of Babu et al. 2021 (arXiv 2111.09296), parsed from the PDF: 128
            languages with ISO code and pre-training hours.
  FLEURS    the google/fleurs repo's own config directory names.
"""
import json, os, re, unicodedata

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(R, exist_ok=True)

FAIRSEQ_URL = ("https://github.com/facebookresearch/fairseq/blob/main/"
               "examples/wav2vec/README.md")
XLSR_PAPER = "https://arxiv.org/abs/2006.13979"
XLSR300_PAPER = "https://arxiv.org/abs/2111.09296"

# ---------------------------------------------------------------- XLSR-53
# Verbatim from the fairseq README's three corpus lines.
MLS = ["Dutch", "English", "French", "German", "Italian", "Polish", "Portuguese", "Spanish"]
COMMONVOICE = ["Arabic", "Basque", "Breton", "Chinese (CN)", "Chinese (HK)", "Chinese (TW)",
    "Chuvash", "Dhivehi", "Dutch", "English", "Esperanto", "Estonian", "French", "German",
    "Hakh-Chin", "Indonesian", "Interlingua", "Irish", "Italian", "Japanese", "Kabyle",
    "Kinyarwanda", "Kyrgyz", "Latvian", "Mongolian", "Persian", "Portuguese", "Russian",
    "Sakha", "Slovenian", "Spanish", "Swedish", "Tamil", "Tatar", "Turkish", "Welsh"]
BABEL = ["Assamese", "Bengali", "Cantonese", "Cebuano", "Georgian", "Haitian", "Kazakh",
    "Kurmanji", "Lao", "Pashto", "Swahili", "Tagalog", "Tamil", "Tok", "Turkish",
    "Vietnamese", "Zulu"]

# ISO-639-1 (or -3 where no -1 exists) for every XLSR-53 language name above.
XLSR53_ISO = {
    "Dutch": "nl", "English": "en", "French": "fr", "German": "de", "Italian": "it",
    "Polish": "pl", "Portuguese": "pt", "Spanish": "es", "Arabic": "ar", "Basque": "eu",
    "Breton": "br", "Chinese (CN)": "zh", "Chinese (HK)": "yue", "Chinese (TW)": "zh-TW",
    "Chuvash": "cv", "Dhivehi": "dv", "Esperanto": "eo", "Estonian": "et",
    "Hakh-Chin": "cnh", "Indonesian": "id", "Interlingua": "ia", "Irish": "ga",
    "Japanese": "ja", "Kabyle": "kab", "Kinyarwanda": "rw", "Kyrgyz": "ky",
    "Latvian": "lv", "Mongolian": "mn", "Persian": "fa", "Russian": "ru", "Sakha": "sah",
    "Slovenian": "sl", "Swedish": "sv", "Tamil": "ta", "Tatar": "tt", "Turkish": "tr",
    "Welsh": "cy", "Assamese": "as", "Bengali": "bn", "Cantonese": "yue", "Cebuano": "ceb",
    "Georgian": "ka", "Haitian": "ht", "Kazakh": "kk", "Kurmanji": "ku", "Lao": "lo",
    "Pashto": "ps", "Swahili": "sw", "Tagalog": "tl", "Tok": "tpi", "Vietnamese": "vi",
    "Zulu": "zu",
}


def build_xlsr53():
    per = {"MLS": MLS, "CommonVoice": COMMONVOICE, "BABEL": BABEL}
    iso = sorted({XLSR53_ISO[n] for lst in per.values() for n in lst})
    names = sorted({n for lst in per.values() for n in lst})
    return {
        "model": "facebook/wav2vec2-large-xlsr-53",
        "advertised_language_count": 53,
        "sources": {"primary": FAIRSEQ_URL, "paper": XLSR_PAPER},
        "note": ("fairseq lists MLS (8) + CommonVoice (36) + BABEL (17) = 61 corpus entries; "
                 "de-duplicating by language yields %d distinct ISO codes, one short of the "
                 "advertised 53. The discrepancy is in the published sources (most likely "
                 "Cantonese vs Chinese (HK) being counted separately); it does not affect "
                 "group assignment, since every language used here is unambiguously in or "
                 "out of the union." % len(iso)),
        "by_corpus": per,
        "iso_codes": iso,
        "names": names,
    }


# ---------------------------------------------------------------- XLS-R 300m
def parse_xlsr300():
    """Table 1 of Babu et al. 2021: name block, then ISO/family block, then hours block,
    repeated once per printed page. Names and hours are contiguous and equal-length, so
    they can be zipped; the ISO/family column is interleaved in the PDF text layer and is
    not relied on here."""
    t = open(f"{D}/xlsr300_paper.txt").read()
    pages = [("Abkhazian", "Kinyarwanda"), ("Kyrgyz", "Uzbek"), ("Vietnamese", "Zulu")]
    langs, hours = [], []
    for first, last in pages:
        i = t.index(first)
        seg = t[i:t.index("Table 1:")] if last == "Zulu" else t[i:]
        lines = [l.strip() for l in seg.split("\n")]
        # names: contiguous run from `first` up to `last`
        start = lines.index(first)
        end = lines.index(last, start)
        names = [l for l in lines[start:end + 1] if l]
        # hours: the first contiguous run of numeric tokens, of the same length, after names
        nums, run = [], []
        for l in lines[end + 1:]:
            if re.fullmatch(r"<?\d+", l):
                run.append(0.5 if l == "<1" else float(l))
            elif run:
                if len(run) == len(names):
                    nums = run; break
                run = []
        if not nums and len(run) == len(names):
            nums = run
        assert len(nums) == len(names), (first, len(names), len(nums))
        langs += names; hours += nums
    assert len(langs) == 128, len(langs)
    return langs, hours


# name -> ISO for the XLS-R table (only the entries we need to reconcile with FLEURS)
XLSR300_ISO = {
    "Abkhazian": "ab", "Afrikaans": "af", "Albanian": "sq", "Amharic": "am", "Arabic": "ar",
    "Armenian": "hy", "Assamese": "as", "Azerbaijani": "az", "Bashkir": "ba", "Basque": "eu",
    "Belarusian": "be", "Bengali": "bn", "Bosnian": "bs", "Breton": "br", "Bulgarian": "bg",
    "Burmese": "my", "Cantonese": "yue", "Catalan": "ca", "Cebuano": "ceb",
    "Central Khmer": "km", "Chinese CN": "zh", "Chinese HK": "zh-HK", "Chinese TW": "zh-TW",
    "Chuvash": "cv", "Croatian": "hr", "Czech": "cs", "Danish": "da", "Divehi": "dv",
    "Dutch": "nl", "English": "en", "Esperanto": "eo", "Estonian": "et", "Faroese": "fo",
    "Finnish": "fi", "French": "fr", "Galician": "gl", "Ganda": "lg", "Georgian": "ka",
    "German": "de", "Greek": "el", "Guarani": "gn", "Gujarati": "gu", "Haitian": "ht",
    "Hakha Chin": "cnh", "Hausa": "ha", "Hawaiian": "haw", "Hebrew": "he", "Hindi": "hi",
    "Hungarian": "hu", "Icelandic": "is", "Indonesian": "id", "Interlingua": "ia",
    "Irish": "ga", "Italian": "it", "Japanese": "ja", "Javanese": "jv", "Kabyle": "kab",
    "Kannada": "kn", "Kazakh": "kk", "Kinyarwanda": "rw", "Kyrgyz": "ky", "Korean": "ko",
    "Kurmanji": "ku", "Lao": "lo", "Latin": "la", "Latvian": "lv", "Lingala": "ln",
    "Lithuanian": "lt", "Luxembourgish": "lb", "Macedonian": "mk", "Malagasy": "mg",
    "Malay": "ms", "Malayalam": "ml", "Maltese": "mt", "Manx": "gv", "Maori": "mi",
    "Marathi": "mr", "Mongolian": "mn", "Nepali": "ne", "Norwegian": "no", "Nynorsk": "nn",
    "Occitan": "oc", "Oriya": "or", "Pashto": "ps", "Persian": "fa", "Polish": "pl",
    "Portuguese": "pt", "Punjabi": "pa", "Romanian": "ro", "Romansh Sursilvan": "rm-sursilv",
    "Romansh Vallader": "rm-vallader", "Russian": "ru", "Sakha": "sah", "Sanskrit": "sa",
    "Scots": "sco", "Serbian": "sr", "Shona": "sn", "Sindhi": "sd", "Sinhala": "si",
    "Slovakian": "sk", "Slovenian": "sl", "Somali": "so", "Sorbian Upper": "hsb",
    "Spanish": "es", "Sundanese": "su", "Swahili": "sw", "Swedish": "sv", "Tagalog": "tl",
    "Tajik": "tg", "Tamil": "ta", "Tatar": "tt", "Telugu": "te", "Thai": "th",
    "Tibetan": "bo", "Tok": "tpi", "Turkish": "tr", "Turkmen": "tk", "Ukrainian": "uk",
    "Urdu": "ur", "Uzbek": "uz", "Vietnamese": "vi", "Votic": "vot", "Waray": "war",
    "Welsh": "cy", "Western Frisian": "fy", "Yiddish": "yi", "Yoruba": "yo", "Zulu": "zu",
}


def build_xlsr300():
    langs, hours = parse_xlsr300()
    missing = [l for l in langs if l not in XLSR300_ISO]
    assert not missing, missing
    rows = [{"name": n, "iso": XLSR300_ISO[n], "pretrain_hours": h}
            for n, h in zip(langs, hours)]
    return {
        "model": "facebook/wav2vec2-xls-r-300m",
        "advertised_language_count": 128,
        "parsed_language_count": len(rows),
        "total_hours_parsed": round(sum(hours)),
        "sources": {"paper": XLSR300_PAPER, "table": "Table 1", "fairseq": FAIRSEQ_URL},
        "languages": rows,
        "iso_codes": sorted({r["iso"] for r in rows}),
    }


# ---------------------------------------------------------------- FLEURS
# FLEURS config -> (display name, ISO-639-1/3 matching the pre-training lists, family)
FLEURS = {
 "af_za":("Afrikaans","af","Indo-European"),      "am_et":("Amharic","am","Afro-Asiatic"),
 "ar_eg":("Arabic","ar","Afro-Asiatic"),          "as_in":("Assamese","as","Indo-European"),
 "ast_es":("Asturian","ast","Indo-European"),     "az_az":("Azerbaijani","az","Turkic"),
 "be_by":("Belarusian","be","Indo-European"),     "bg_bg":("Bulgarian","bg","Indo-European"),
 "bn_in":("Bengali","bn","Indo-European"),        "bs_ba":("Bosnian","bs","Indo-European"),
 "ca_es":("Catalan","ca","Indo-European"),        "ceb_ph":("Cebuano","ceb","Austronesian"),
 "ckb_iq":("Sorani Kurdish","ckb","Indo-European"),"cmn_hans_cn":("Mandarin","zh","Sino-Tibetan"),
 "cs_cz":("Czech","cs","Indo-European"),          "cy_gb":("Welsh","cy","Indo-European"),
 "da_dk":("Danish","da","Indo-European"),         "de_de":("German","de","Indo-European"),
 "el_gr":("Greek","el","Indo-European"),          "en_us":("English","en","Indo-European"),
 "es_419":("Spanish","es","Indo-European"),       "et_ee":("Estonian","et","Uralic"),
 "fa_ir":("Persian","fa","Indo-European"),        "ff_sn":("Fula","ff","Atlantic-Congo"),
 "fi_fi":("Finnish","fi","Uralic"),               "fil_ph":("Filipino","tl","Austronesian"),
 "fr_fr":("French","fr","Indo-European"),         "ga_ie":("Irish","ga","Indo-European"),
 "gl_es":("Galician","gl","Indo-European"),       "gu_in":("Gujarati","gu","Indo-European"),
 "ha_ng":("Hausa","ha","Afro-Asiatic"),           "he_il":("Hebrew","he","Afro-Asiatic"),
 "hi_in":("Hindi","hi","Indo-European"),          "hr_hr":("Croatian","hr","Indo-European"),
 "hu_hu":("Hungarian","hu","Uralic"),             "hy_am":("Armenian","hy","Indo-European"),
 "id_id":("Indonesian","id","Austronesian"),      "ig_ng":("Igbo","ig","Atlantic-Congo"),
 "is_is":("Icelandic","is","Indo-European"),      "it_it":("Italian","it","Indo-European"),
 "ja_jp":("Japanese","ja","Japonic"),             "jv_id":("Javanese","jv","Austronesian"),
 "ka_ge":("Georgian","ka","Kartvelian"),          "kam_ke":("Kamba","kam","Atlantic-Congo"),
 "kea_cv":("Kabuverdianu","kea","Creole"),        "kk_kz":("Kazakh","kk","Turkic"),
 "km_kh":("Khmer","km","Austro-Asiatic"),         "kn_in":("Kannada","kn","Dravidian"),
 "ko_kr":("Korean","ko","Koreanic"),              "ky_kg":("Kyrgyz","ky","Turkic"),
 "lb_lu":("Luxembourgish","lb","Indo-European"),  "lg_ug":("Ganda","lg","Atlantic-Congo"),
 "ln_cd":("Lingala","ln","Atlantic-Congo"),       "lo_la":("Lao","lo","Kra-Dai"),
 "lt_lt":("Lithuanian","lt","Indo-European"),     "luo_ke":("Luo","luo","Nilotic"),
 "lv_lv":("Latvian","lv","Indo-European"),        "mi_nz":("Maori","mi","Austronesian"),
 "mk_mk":("Macedonian","mk","Indo-European"),     "ml_in":("Malayalam","ml","Dravidian"),
 "mn_mn":("Mongolian","mn","Mongolic"),           "mr_in":("Marathi","mr","Indo-European"),
 "ms_my":("Malay","ms","Austronesian"),           "mt_mt":("Maltese","mt","Afro-Asiatic"),
 "my_mm":("Burmese","my","Sino-Tibetan"),         "nb_no":("Norwegian","no","Indo-European"),
 "ne_np":("Nepali","ne","Indo-European"),         "nl_nl":("Dutch","nl","Indo-European"),
 "nso_za":("Northern Sotho","nso","Atlantic-Congo"),"ny_mw":("Nyanja","ny","Atlantic-Congo"),
 "oc_fr":("Occitan","oc","Indo-European"),        "om_et":("Oromo","om","Afro-Asiatic"),
 "or_in":("Odia","or","Indo-European"),           "pa_in":("Punjabi","pa","Indo-European"),
 "pl_pl":("Polish","pl","Indo-European"),         "ps_af":("Pashto","ps","Indo-European"),
 "pt_br":("Portuguese","pt","Indo-European"),     "ro_ro":("Romanian","ro","Indo-European"),
 "ru_ru":("Russian","ru","Indo-European"),        "sd_in":("Sindhi","sd","Indo-European"),
 "sk_sk":("Slovak","sk","Indo-European"),         "sl_si":("Slovenian","sl","Indo-European"),
 "sn_zw":("Shona","sn","Atlantic-Congo"),         "so_so":("Somali","so","Afro-Asiatic"),
 "sr_rs":("Serbian","sr","Indo-European"),        "sv_se":("Swedish","sv","Indo-European"),
 "sw_ke":("Swahili","sw","Atlantic-Congo"),       "ta_in":("Tamil","ta","Dravidian"),
 "te_in":("Telugu","te","Dravidian"),             "tg_tj":("Tajik","tg","Indo-European"),
 "th_th":("Thai","th","Kra-Dai"),                 "tr_tr":("Turkish","tr","Turkic"),
 "uk_ua":("Ukrainian","uk","Indo-European"),      "umb_ao":("Umbundu","umb","Atlantic-Congo"),
 "ur_pk":("Urdu","ur","Indo-European"),           "uz_uz":("Uzbek","uz","Turkic"),
 "vi_vn":("Vietnamese","vi","Austro-Asiatic"),    "wo_sn":("Wolof","wo","Atlantic-Congo"),
 "xh_za":("Xhosa","xh","Atlantic-Congo"),         "yo_ng":("Yoruba","yo","Atlantic-Congo"),
 "yue_hant_hk":("Cantonese","yue","Sino-Tibetan"),"zu_za":("Zulu","zu","Atlantic-Congo"),
}


def main():
    x53 = build_xlsr53()
    x300 = build_xlsr300()
    json.dump(x53, open(f"{R}/xlsr53_languages.json", "w"), indent=1, ensure_ascii=False)
    json.dump(x300, open(f"{R}/xlsr_300m_languages.json", "w"), indent=1, ensure_ascii=False)

    cfgs = json.load(open(f"{D}/fleurs_configs.json"))
    assert set(cfgs) == set(FLEURS), set(cfgs) ^ set(FLEURS)
    seen53 = set(x53["iso_codes"])
    hours300 = {r["iso"]: r["pretrain_hours"] for r in x300["languages"]}
    rows = []
    for cfg, (name, iso, fam) in sorted(FLEURS.items()):
        rows.append(dict(config=cfg, name=name, iso=iso, family=fam,
                         in_xlsr53=iso in seen53,
                         in_xlsr300=iso in hours300,
                         xlsr300_hours=hours300.get(iso)))
    json.dump(rows, open(f"{R}/fleurs_languages.json", "w"), indent=1, ensure_ascii=False)

    print(f"XLSR-53   : {len(x53['iso_codes'])} distinct ISO codes (advertised 53)")
    print(f"XLS-R 300m: {x300['parsed_language_count']} languages, "
          f"{x300['total_hours_parsed']:,} h parsed (paper says ~436k)")
    print(f"FLEURS    : {len(rows)} languages")
    print(f"  FLEURS in XLSR-53   (SEEN pool)  : {sum(r['in_xlsr53'] for r in rows)}")
    print(f"  FLEURS not in XLSR-53 (UNSEEN)   : {sum(not r['in_xlsr53'] for r in rows)}")
    print(f"  FLEURS in XLS-R 300m             : {sum(r['in_xlsr300'] for r in rows)}")


if __name__ == "__main__":
    main()
