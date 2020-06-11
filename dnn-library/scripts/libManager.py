#!/usr/bin/env python3
import os
import re
import argparse
import sys
from string import Template
import functools

from openpyxl import load_workbook, Workbook
from openpyxl.styles import colors, PatternFill, Color, Font
from openpyxl.utils import get_column_letter

class OperatorsEnum:
    def __init__(self, hostswdir):
        fname = os.path.join ( hostswdir, 'Neuralizer/inc/IOperators.inc');
        regexp = re.compile(r'\s*IOPERATOR_ENUM\((.*)\)\s*')
        self.__enum = []
        with open(fname) as f:
            for line in f:
                m = regexp.match( line )
                if m:
                    self.__enum.append("ET_" + m.group(1))

    def get(self):
        return self.__enum
        
class LibManagerSheet:
    ################################################################################
    # constants
    ################################################################################
    headerKeys = ("Operator", "nrOutTensors", "nrInTensors", "members", "templateElk","extraImpl", "implSel")
    headerFill = PatternFill(fgColor = "DDDDDD", fill_type = "solid")
    #    operatorFill = [PatternFill(fgColor = "3FBE7E", fill_type = "solid"),
    #                    PatternFill(fgColor = "5F9E9F", fill_type = "solid") ]
    operatorFill = [None]

    def __init__(self, spreadsheet, hostswdir = None, glowdir = None):

        enum = OperatorsEnum(hostswdir)
        self.__enum = enum.get()
        
        try:
            self.__wb = load_workbook(spreadsheet)            
            self._existing = True
        except FileNotFoundError:
            self._existing = False

        if self._existing:
            self.__codeGen(hostswdir)
        else:
            self.__xlsGen(spreadsheet, hostswdir, glowdir)

    ################################################################################
    # methods to create the spread sheet skeleton
    ################################################################################

    def __xlsGen(self, spreadsheet, hostswdir, glowdir):
        # parse instruction generator
        insts = InstructionGenParser(glowdir)

        #find implementations in dnn_lib dir
        implementations = self.__findImplementations(hostswdir + "/dnn_lib/include/inlining")

        
        print ("Excel not found: creating. Edit if necessary and rerun.")
        self.__row = 1
        self.__wb = Workbook()
        self.__ws = self.__wb.active
        self.__ws.title = "LibManager"
        
        # header
        self.__addRow(LibManagerSheet.headerKeys, LibManagerSheet.headerFill)

        # add operators
        for op in sorted(insts.get_all()):
            conf = insts.get(op)
            
            # count tensor operands            
            nrIn = 0
            nrOut = 0
            for i in conf['operands']:
                if i['kind'] == 'OperandKind::Out' or i['kind'] == 'OperandKind::InOut':
                    nrOut+=1
                elif i['kind'] == 'OperandKind::In':
                    nrIn+=1
                else:
                    print("in %s, tensor operand kind %s not supported: %s will be ignored" % (op, i['kind'], i["name"]), file = sys.stderr)

            # list non tensor operands
            members = [ i['name'] for i in conf['members']]

            template_params = "" #empty, will have to be added manually in the spreasheet
            
            if op not in implementations:
                print("no implementations found for %s: skipping" % op, file = sys.stderr)
                continue
                
            extraImpl = [ k for k in implementations[op] ]
                
            values = [ op, nrOut, nrIn,
                       ', '.join(members),
                       template_params,
                       ', '.join(extraImpl),
                       "default"]
            
            self.__addRow(values, LibManagerSheet.operatorFill[self.__row % len(LibManagerSheet.operatorFill) ])

            
        # and adjust width to fit contents
        self.__adjustColumnWidth()

        # and finally save
        self.__wb.save(filename = spreadsheet)

    def __addRow(self, values, fill):
        for i, val in enumerate (values):
            cell = self.__ws.cell(self.__row, 1 + i, val)
            if fill != None:
                cell.fill = fill
        self.__row+=1

    def __adjustColumnWidth(self):
        for column_cells in self.__ws.columns:
            col = column_cells[0].column
            col_letter = get_column_letter(col)
            fit = 6 +  max(len(str(cell.value)) for cell in column_cells)                
            self.__ws.column_dimensions[col_letter].width = fit


    def __findImplementations(self, baseDir):
        implementations={}
        reHeader = re.compile(r'.*\.h')
        reImpl = re.compile(r'.*void.*fwdLib(.*)Inst(.*)\(.*')
        eltWiseImpl = re.compile(r'\s*EltWiseInst\((.*), .*')
        
        files = [os.path.join(baseDir, f) for f in os.listdir(baseDir) if os.path.isfile(os.path.join(baseDir, f)) and reHeader.match(f)]        
        for fname in files:
            found = 0
            with open(fname) as f:
                for line in f:
                    m = reImpl.match(line)
                    if m:
                        op = m.group(1)
                        details = m.group(2)
                        if op not in implementations:
                            implementations[op] = []
                        if len(details) > 0:
                            implementations[op].append(details)
                        continue
                    # special way elementwise instructions are defined
                    m = eltWiseImpl.match(line)
                    if m:
                        op = m.group(1)
                        implementations[op] = ['Threaded', 'Vectorized']
                        
        return implementations
    
    ################################################################################
    #  Methods to retrieve information from the sheet
    ################################################################################
        
    def __codeGen(self, hostswdir):
        # load configuration from spreadsheet
        self.loadSheet()

        # generate table for header file
        table = [ self.tableEntry(i) for i in self.__enum ] 
        tableStr = self.formatTable(table)
       
        # check for entries in the header file that have not been used
        missing = [i for i in self.__configs if not self.__configs[i]["gen"]]
        for i in missing:
            print("Spreadhseet row for %s not used" % i, file = sys.stderr)

        # and output inplace
        self.output(hostswdir, tableStr)
                
    def loadSheet(self):
        ws = self.__wb.active
        self.__configs = {}
        skipHeader = True
        for row in ws.rows:
            if skipHeader:
                skipHeader = False
                continue

            operator = { "gen": False}
            for h,cell in zip(LibManagerSheet.headerKeys, row):
                operator[h] = cell.value
                
            enum = "ET_" + operator["Operator"].lower()
            self.__configs[enum] = operator

    def tableEntry(self, op):
        if op in self.__configs:
            conf = self.__configs[op];
            conf["gen"] = True
            members = []
            versions = []
            template = 0
            if conf["members"]:
                members = ["mb" + i.replace(" ", "") for i in conf["members"].split(',')]
            if conf["extraImpl"]:
                versions = ['"' + i.replace(" ", "") + '"' for i in conf["extraImpl"].split(',')]
            if conf["templateElk"] == None:
                raise Exception("empty tenplate definition for %s. Use NONE if the fnc doesn't use templates" % op)
            elif conf["templateElk"] != "NONE":
                template = functools.reduce( lambda a,b : a | (1 << int(b)), str(conf["templateElk"]).split(','), 0) ;

            return { "enum": op,
                     "name" : conf["Operator"],
                     "nrOutputTensors": conf["nrOutTensors"],
                     "nrInputTensors": conf["nrInTensors"],
                     "members": "{%s}" % ", ".join(members),
                     "template": template,
                     "versions":  "{%s}" % ", ".join(versions)
            }
                     #TODO: add implSel
        else:
            print("WARN: Could not find spreadsheet row for %s" % op, file = sys.stderr)
            return {"enum": op,
                    "name" : "notImplemented",
                    "nrOutputTensors": 0,
                    "nrInputTensors": 0,
                    "members": "{}",
                    "template": 0,
                    "versions": "{}"
            }
                       #TODO: add implSel


    def formatTable(self, table):
        s = Template('''     /**** $enum ****/
     { "$name", // name
       $nrOutputTensors, // # outs
       $nrInputTensors,  // # ins
       $members, // members
       $template, // template param mask
       $versions // impl versions
     }''')
        entries = [ s.substitute(e) for e in table]        
        return ",\n".join(entries)

    def output(self, hostswdir, tableStr):
        contents = []
        startMark = re.compile(r'\s*// INSTR_CONFIG_TABLE_BEGIN')
        endMark = re.compile(r'\s*// INSTR_CONFIG_TABLE_END')
        fname = os.path.join ( hostswdir, 'dnn_lib/include/LibApi.h');
        found = False
        written = False
        with open(fname) as f:
            for line in f:
                if not found:
                    contents.append(line)
                    if startMark.match(line):
                        contents.append(tableStr + "\n")
                        written = True
                        found = True
                else:
                    if endMark.match(line):
                        contents.append(line)
                        found = False

        if found:
            raise Exception("didn't not find end of table in %s" % fname)
        if not written:
            raise Exception("didn't not find start of table in %s" % fname)

        with open(fname, "w") as f:
            f.writelines(contents)

        
                       
if __name__ == "__main__":
    # parse command line options
    parser = argparse.ArgumentParser("Create Operator test")
    parser.add_argument("--hostsw-dir", help="host-sw root dir", required = True) 
    parser.add_argument("--glow-dir", help="esperanto-glow root dir")
    parser.add_argument("--excel", help="Excel file to use", required = True)
    args = parser.parse_args()
    sys.path.append(os.path.join (args.hostsw_dir, 'scripts/testing/operatorTests/'))
    from instructionGenParser import InstructionGenParser

    
    sheet = LibManagerSheet(args.excel, args.hostsw_dir, args.glow_dir)

