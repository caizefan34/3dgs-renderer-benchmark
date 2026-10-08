src = '''def _param(self, name):
    return {"xyz": self.model._xyz, "shs": self.model._shs,
            "opacity": self.model._opacity,
            "scaling": self.model._scaling,
            "rotation": self.model._rotation}[name]
'''
try:
    compile(src, "x", "exec")
    print("isolated OK")
except SyntaxError as e:
    print("isolated FAIL:", e.lineno, e.offset, e.msg)
