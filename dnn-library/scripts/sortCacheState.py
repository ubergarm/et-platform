#!/usr/bin/env python3

from openpyxl import load_workbook, Workbook
import argparse

parser = argparse.ArgumentParser("Sorts excel sheet alphabetically")
parser.add_argument("spreadsheet", help="Excel file to use with the cache state")
args = parser.parse_args()

wb = load_workbook(args.spreadsheet)
wb._sheets.sort(key=lambda ws: ws.title)
wb.save(args.spreadsheet)
