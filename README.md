# INSA-Zadanie

---

**Meno a priezvisko:** Samuel Titko

**Odbor:** Inteligentné systémy



`Zadanie_1`
- práca s jupyter notebookom

`Zadanie_2`
- vlastný objektovo orientovaný pipeline

`Zadanie_3`
- pipeline knižnice sklearn

`Zadanie_4`
- transformácia sklearn pipeline z výskumného do produkčného kódu

```
+ INSA-Zadanie\
|
├──+ Zadanie_1\
|  ├──+ dataset\
|  |  ├── dataset.csv
|  |  ├── gender_submission.csv
|  |  ├── train.csv
|  |  └── test.csv
|  ├── merge.py
|  └── main.ipynb
├──+ Zadanie_2\
|  ├──+ dataset\
|  |  └── dataset.csv
|  ├── config.py
|  ├── main.py
|  └── pipeline.py
├──+ Zadanie_3\
|  ├──+ dataset\
|  |  └── dataset.csv
|  ├── config.py
|  ├── main.py
|  └── transformers.py
├──+ Zadanie_4\
|  ├──+ pipeline\
|  |  ├──+ model\
|  |  |  ├──+ dataset\
|  |  |  |  ├── test.csv
|  |  |  |  └── train.csv
|  |  |  ├──+ trained_models\
|  |  |  |  └── ...
|  |  |  ├── __init__.py
|  |  |  ├── config.py
|  |  |  ├── manager.py
|  |  |  ├── pipeline.py
|  |  |  ├── predict.py
|  |  |  ├── test.py
|  |  |  ├── train.py
|  |  |  ├── transformers.py
|  |  |  ├── validation.py
|  |  |  └── VERSION
|  |  ├──+ tests\
|  |  |  ├── __init__.py
|  |  |  └── unit_tests.py
|  |  ├── MANIFEST.in
|  |  ├── requirements.txt
|  |  ├── run_tests.py
|  |  ├── setup.py
|  |  └── tox.ini
|  └──+ rest\
|     ├──+ api\
|     |  └── app.py
|     └──+ tests\
|       └── unit_tests.py
├── .gitignore
├── Main.py
└── README.md
```

## Zadanie_4
Postup setupu `pipeline`:
1. `cd ./pipeline`,
2. `python -m venv .virenv`
3. `.\.virenv\Scripts\pip.exe install tox`
4. `.\.virenv\Scripts\pip.exe install -r requirements.txt`
5. `.\.virenv\Scripts\python.exe run_tests.py`

Postup setupu `rest`:
1. `cd ./rest`,
2. `python -m venv .virenv`
3. `.\.virenv\Scripts\pip.exe install pytest`
4. `.\.virenv\Scripts\pip.exe install -r requirements.txt`
5. `.\.virenv\Scripts\python.exe .\tests\unit_tests.py`
