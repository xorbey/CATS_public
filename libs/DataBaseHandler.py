from pymongo import MongoClient
import datetime
import pandas as pd

class DatabaseHandler:

    def __init__(self, url:str, port:int, robot:str, traj:str):
        self.client = MongoClient(url, port)
        self.mydb = self.client[robot]  # name of data base
        self.featcollection = self.mydb[traj]
        self.predcollection = self.mydb[traj + "_predictions"]

    def writeData(self, values):
        """
        Pushes data into the database
        :param values: list of dictionaries of format {"seg": *stringvalue*, "sensor": *stringvalue*, "key":*stringvalue*
        , "features":{"featureName":*featureValue*}}
        :return:
        """
        #TODO check with transform component

        self.featcollection.insert_many(values)

    def getData(self, seg, sensorsWithFeatures, start, stop):
        sensorkeys = list(sensorsWithFeatures.keys())
        searchdic = {"key":{"$gt": start, "$lt": stop}, "seg":seg, "sensor": {"$in": sensorkeys}}
        data = [e for e in self.featcollection.find(searchdic)]
        keys = [e["key"] for e in data]
        dataDf = pd.DataFrame()
        for key in keys:
            dataKey = [e for e in data if e["key"] == key]
            dataFeatures = {}
            for dat in dataKey:
                dataFeatures.update({dat["sensor"]+"_"+f:dat['features'][f] for f in dat["features"]})
            df = pd.DataFrame(dataFeatures, index = [key])
            dataDf = dataDf.append(df)
        return dataDf

    def getAllData(self, seg, sensorsWithFeatures):
        sensorkeys = list(sensorsWithFeatures.keys())
        searchdic = {"seg":seg, "sensor": {"$in": sensorkeys}}
        data = [e for e in self.featcollection.find(searchdic)]
        keys = [e["key"] for e in data]
        dataDf = pd.DataFrame()
        for key in keys:
            dataKey = [e for e in data if e["key"] == key]
            dataFeatures = {}
            for dat in dataKey:
                dataFeatures.update({dat["sensor"]+"_"+f:dat['features'][f] for f in dat["features"]})
            df = pd.DataFrame(dataFeatures, index = [key])
            dataDf = dataDf.append(df)
        return dataDf

    def writePrediction(self, seg, key, sensorsWithFeatures, prediction):
        mydict = {"key":key, "seg": seg, "sensorsWithFeatures": sensorsWithFeatures, "prediction": prediction}
        self.predcollection.insert_one(mydict)





