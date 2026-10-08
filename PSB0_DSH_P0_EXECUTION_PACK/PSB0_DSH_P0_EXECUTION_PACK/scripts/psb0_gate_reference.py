def gate(operation_reduction, pred_e2e):
    return operation_reduction >= .20 and pred_e2e >= 1.05
if __name__ == "__main__":
    assert gate(.25,1.06)
    print("PSB0 gate reference self-test PASS")
