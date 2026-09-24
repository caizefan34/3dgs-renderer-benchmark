import sys
for mod in ["sklearn", "scipy"]:
    try:
        m = __import__(mod)
        print(f"{mod}: {m.__version__}")
    except ImportError:
        print(f"{mod}: NOT INSTALLED")
