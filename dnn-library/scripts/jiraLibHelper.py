from libManager import LibManagerSheet
from jira import JIRA
from jira.resources import User
import os


# create derived classes implementing the following functions:
#   filter(self, conf) => receives config from excel. returns true if jira has to be created
#   format(self, conf) => returns tuple (summary, description) to fill jira information
#
# To use, you might need to "pip install jira"
#
# You will also need to create a jira api token, and store in a file with the following format:
#   user:your@email.com
#   token:XXXXX
# the default path for the token credentials is ~/.jira_api_token

class JiraLibHelper(LibManagerSheet):
    defaultServer = 'https://esperantotech.atlassian.net/'
    defaultTokenFile = '.jira_api_token'


    # spreadsheet = libManager excel
    # swplatform_root = repo root dir
    # jiraConf: dict with the following keys:
    #    assignee(e.g. e-mail)
    #    components (e.g. ET-Libraries)
    #    epic (SW-XXX)
    #    issueType(Task, Epic, Feature, Bug...)
    #    project (SW)

    def __init__(self, spreadsheet, swplatform_root, jiraConf, server = defaultServer, token = defaultTokenFile, dryrun = False):
        # spreadsheet related
        hostswdir = os.path.join (swplatform_root,'host-software/host-sw/')
        glowdir = os.path.join (swplatform_root,'host-software/glow/')
        super().__init__(spreadsheet,  None, hostswdir , glowdir, True)

        #jira server related
        self.__jiraAuth = self.__getApiToken(token)
        self.__jiraServer = server
        self.__jiraConf = jiraConf
        self.__dryrun = dryrun
        self.__userIds = {}
        self.__epicIds = {}


    def __getApiToken(self, filename):
        auth = {}
        filename = os.path.join(os.path.expanduser("~"), filename)
        with open(filename, "r") as f:
            for line in f:
                data = line.split(':')
                if len(data) == 2:
                    auth[data[0]] = data[1].strip('\n')
        if 'user' not in auth or 'token' not in auth:
            raise Exception("api token file with incorrect format")
        
        return (auth['user'], auth['token']) 
        

    def go(self):
        # filter operators
        operators = [op for op in self._configs if self.filter(self._configs[op])]

        # create jira object
        self.__jira = JIRA( basic_auth=self.__jiraAuth,
                           options={'server': self.__jiraServer } )

        # loop and create
        for op in operators:
            (summary, description) = self.format(self._configs[op])
            self.create(summary, description,
                        self.__jiraConf['assignee'],
                        self.__jiraConf['components'],
                        self.__jiraConf['epic'],
                        self.__jiraConf['project'],
                        self.__jiraConf['issueType'])

    # returns True/False depending on whether to generate a jira for this operator or not
    def filter(self, conf):
        raise NotImplementedError()

    # returns tuple (summary, description)
    def format(self, conf):
        raise NotImplementedError()

    #issueTypes: Task, Epic, Feature, Bug...
    def create(self, summary, description, assignee, components, epic = None, project='SW', issueType = 'Task'):
        # check there is not a jira with the same summary
        similar = self.__jira.search_issues(f'summary ~ "\\"{summary}\\""')
        for i in similar:
            if i.fields.summary == summary:
                print (f"WARN: Skipping. There's already an existing issue ({i.key}) with summary=\"{summary}\"")
                return
        
        # get epic Id
        if epic != None:
            epicId = self.__getEpicId(epic)

        # create components array
        if not isinstance(components, list):
            comp = [{'name': components}]
        else:
            comp = [ {'name': c} for c in components]


        assigneeId = self.__getUserId(assignee)
            
        # prepare issue options
        issue_dict = {
            'summary': summary,
            'description': description,
            'project': {'key': project},
            'assignee': {'accountId': assigneeId},
            'issuetype': {'name': issueType},
            'components': comp
        }

        # create
        if self.__dryrun:
            print (f""""############# the following jira would be created #############
summary: {summary}
assignee: {assignee} [id={assigneeId}]
type={issueType}, components:{comp}, project={project}, epic={epic}
description:
{description}""")
        else:
            new_issue = self.__jira.create_issue(fields=issue_dict)
            print ("Created %s: %s" % (new_issue.key, summary))

        # add to epic
        if epic != None and not self.__dryrun:
            try:
                self.__jira.add_issues_to_epic(epicId, [new_issue.key])
            except:
                print("*** WARN: could not add to EPIC")

    def __getEpicId(self, key):
        if key in self.__epicIds:
            return self.__epicIds[key]
        issue = self.__jira.issue(key)
        self.__epicIds[key] = issue.id
        return issue.id
    
    def __getUserId(self, user):
        if user in self.__userIds:
            return self.__userIds[user]
        if user == None:
            return None
        u = self.__jira._fetch_pages(User, None, "user/search", 0, 2, {"query": user})
        if len(u) == 0:
            raise Exception("Cannot find assignee %s" % user)
        elif len(u) > 1:
            raise Exception("Assignee string %s matches more than 1 user" % user)
        else:
            self.__userIds[user] = u[0].accountId
            self.__userIds[user]
        

    
