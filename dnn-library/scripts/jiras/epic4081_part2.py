#!/usr/bin/env python3


# takes all operators with more than 1 version and without custom impl selector
# this means that the last version is supposed to be generic, so other implementations
# can be removed. Adds a jira to epic 4081 for this purpose.


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
            extra = len(conf["extraImpl"].split(','))
            if extra > 0 and conf["implSel"] != "custom":
                return True
        else:
            return False



    def format(self, conf):
        versions = ["baseImpl"] + [i.replace(" ", "") for i in conf["extraImpl"].split(',')]
        summary=f"Operator {conf['Operator']} has multiple implementations and no custom impl selector"
        implList ="\n".join(["* " + v for v in versions])
        desc="""%s operator has the following implementations:
%s

Because it doesn't have a custom implementation selector (no conditions to fall back to a least optimized version), it can be assumed that all current versions are generic. This means that the least optimized implementations can be removed to simplify testing (less test cases).

So,
* remove all the implementations but the last one (%s), which has to be renamed so that it is the base implementation.
* update the libManager.xlsx: clear the extraImpl column
* update cacheState.xlsx accordingly: remove references to removed implementations
* rerun the script to regenerate the libApi
""" % (conf["Operator"], implList, versions[-1])

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
