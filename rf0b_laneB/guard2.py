p = "rf0b_trainer.py"
s = open(p, encoding="utf-8").read()
old = '''    @contextmanager
    def _fastgs_scoring_guard():
        was_pending = _HIGS_DYNAMIC_SCENE._pending_backward
        try:
            yield
        finally:
            _HIGS_DYNAMIC_SCENE._pending_backward = was_pending
'''
new = '''    @contextmanager
    def _fastgs_scoring_guard():
        handle = _HIGS_DYNAMIC_SCENE.renderer_handle
        was_pending = handle._pending_backward if handle is not None else False
        try:
            yield
        finally:
            if handle is not None:
                handle._pending_backward = was_pending
'''
assert old in s, "guard block not found"
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("GUARD FIX APPLIED")