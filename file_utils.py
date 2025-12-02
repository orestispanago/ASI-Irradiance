import csv


def list_to_csv(the_list, fname):
    with open(fname, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(the_list)
