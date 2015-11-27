import csv

def unicode_reader(data):
	reader = csv.reader(data)
	for row in reader:
		yield [cell.encode('utf-8') for cell in row]