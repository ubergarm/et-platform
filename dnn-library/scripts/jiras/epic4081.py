#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append('..')
from jiraLibHelper import JiraLibHelper

class Epic4081(JiraLibHelper):
    def __init__(self, spreadsheet, swplatform_root, jiraConf, dryrun):
        super().__init__(spreadsheet, swplatform_root, jiraConf, dryrun = dryrun)

    def filter(self, conf):
        if conf["extraImpl"]:
            return len(conf["extraImpl"].split(',')) > 1 # more than 1 extra implementation
        else:
            return False

        return len(self.__versions) > 2

    def format(self, conf):
        versions = ["generic(scalar)"] + [i.replace(" ", "") for i in conf["extraImpl"].split(',')]
        summary="Remove least optimized implementations for the " + conf["Operator"] + " operator"
        implList ="\n".join(["* " + v for v in versions])
        desc="""%s operator has the following implementations:
%s

Remove the least optimized generic versions, and keep at most one non-generic optimized implementation
""" % (conf["Operator"], implList)

        return (summary, desc)

if __name__ == "__main__":
    # parse command line options
    parser = argparse.ArgumentParser("Create libApi tables")
    parser.add_argument("--swplatform-root", help="sw-platform root dir", required = True) 
    parser.add_argument("--excel", help="Excel file to use", required = True)
    parser.add_argument("-n","--dry-run", help="Do not actually create any jiras", action='store_true' )
    args = parser.parse_args()
    
    sys.path.append(os.path.join (args.swplatform_root, 'scripts/testing/operatorTests/'))
    from instructionGenParser import InstructionGenParser

    jiraConfig = {
        'assignee': 'sebastia.tortella@esperantotech.com',
        'components': 'ET-Libraries',
        'epic': 'SW-4081',
        'issueType': 'Task',
        'project':'SW'
    }
    e = Epic4081(args.excel, args.swplatform_root, jiraConfig, args.dry_run)
    e.go()
