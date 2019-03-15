#!/usr/bin/env python
# encoding: utf-8

import json
import tweepy


# Authentication details. To  obtain these visit dev.twitter.com
consumer_key = "7WUpCBtaEaFRisyVpZYQjGFcX"
consumer_secret = "zHeGyAGVxjId2SIl0EiJQ4uhOu6xTzplDeGWFKsxBKWi0bpqnN"
access_token = "907813368693612544-hMsBTuN8DnmY4bGn2GuR2G5EP3qDqQ0"
access_token_secret = "YK1CptmhK2walKEep7WbDA3FljDbkgSttccXNfqEGc1pu"

# This is the listener, resposible for receiving data
class StdOutListener(tweepy.StreamListener):
    def on_data(self, data):
	
          # Parsing 
		
        decoded = json.loads(data)
	tweet = decoded["text"]
        #open a file to store the status objects
        file = open('test.json', 'a')
        #write json to file
	file.write('\n')
        json.dump(tweet,file,sort_keys = True,indent = 4)
        #show progress
        print "Writing tweets to file,CTRL+C to terminate the program"
       
        return True

    def on_error(self, status):
        print status

if __name__ == '__main__':
    l = StdOutListener()
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    
    stream = tweepy.Stream(auth, l)
    #Hashtag to stream
    stream.filter(track=["#law"])