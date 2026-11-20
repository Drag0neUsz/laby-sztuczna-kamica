import matplotlib.pyplot as plt

def plot_errors(epochs, train_errors, test_errors, h, filename="wykres_bledu.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_errors, label="tr_set error", color="blue")
    plt.plot(epochs, test_errors, label="test_set error", color="orange")
    plt.xlabel("Epoka")
    plt.ylabel("Błąd epoki")
    plt.title(f"Zmiana błędu na tr_set i test_set (h={h})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def format_ascii_cm(epoch, tr, te):
    return (
        f"\nEpoka {epoch}:\n"
        f"           TRAIN SET                              TEST SET\n"
        f"          Pred 1    Pred 0                       Pred 1    Pred 0\n"
        f"        +---------+---------+                  +---------+---------+\n"
        f"Akt 1   | TP:{tr[0]:<4} | FN:{tr[2]:<4} |          Akt 1   | TP:{te[0]:<4} | FN:{te[2]:<4} |\n"
        f"        +---------+---------+                  +---------+---------+\n"
        f"Akt 0   | FP:{tr[1]:<4} | TN:{tr[3]:<4} |          Akt 0   | FP:{te[1]:<4} | TN:{te[3]:<4} |\n"
        f"        +---------+---------+                  +---------+---------+\n"
    )

