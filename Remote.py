from utils import *
from dataManipulation import *

from os.path import sep
import paramiko



def runRemoteServer(url, uname, passw, trainStack, trainLabels, configToUse, submissionScriptString, folderToUse, pytorchFolder, submissionCommand):
	client = paramiko.SSHClient()
	client.load_system_host_keys()
	client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	client.connect(url, username=uname, password=passw)
	scp = SCPClient(client.get_transport())
	scp.put(trainStack, folderToUse + sep + 'trainImages.tif')
	scp.put(trainLabels, folderToUse + sep + 'trainLabels.tif')
	scp.put(configToUse, folderToUse + sep + 'config.yaml')
	with open('submissionScript.sb','w') as outFile:
		outFile.write(submissionScriptString)
	scp.put(submissionScript, folderToUse + sep + 'submissionScript.sb')
	scp.close()
	stdin, stdout, stderr = client.exec_command(submissionCommand)
	output = stdout.read().decode().strip()
	stdin.close()
	stdout.close()
	stderr.close()
	client.close()
	return output

def getRemoteFile(url, uname, passw, filenameToGet, filenameToStore):
	client = paramiko.SSHClient()
	client.load_system_host_keys()
	client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	client.connect(url, username=uname, password=passw)
	scp = SCPClient(client.get_transport())
	scp.get(filenameToGet, filenameToStore)
	scp.close()
	client.close()

def checkStatusRemoteServer(url, name, passw, jobOutputFile):
	client = paramiko.SSHClient()
	client.load_system_host_keys()
	client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	client.connect(url, username=uname, password=passw)
	command = 'cat ' + jobOutputFile
	stdin, stdout, stderr = client.exec_command(command)
	outputLines = stdout.readlines()
	stdin.close()
	stdout.close()
	stderr.close()
	fullOutput = ''
	for line in outputLines:
		fullOutput += line.strip() + '\n'
	client.close()
	return fullOutput

def getSubmissionScriptAsString(template, memory, time, config, outputDirectory):
	file = open(template,'r')
	stringTemplate = file.read()
	file.close()
	stringTemplate.replace('{memory}',memory)
	stringTemplate.replace('{time}',time)
	stringTemplate.replace('{config}',config)
	stringTemplate.replace('{outputFileDir}',outputDirectory)
	return stringTemplate