import os

config = {}

config['CREATE_ANALYSIS'] = []

if os.getenv('DEBUG', "true").lower() == "true":
    config['DEBUG']=True
    config['CONNECTION_URI']='mongodb://root:s3cr3t@localhost:27017'
    config['CONNECTION_DB']='ganabosques'

else:
    config['DEBUG']=False
    config['CONNECTION_URI']=os.getenv('CONNECTION_URI')
    config['CONNECTION_DB']=os.getenv('CONNECTION_DB')

