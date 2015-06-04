import xml.dom.minidom
import copy
import guid
import math
import os
import operator
from compiler.ast import flatten

# A couple contants
CONTINUOUS = 0
DISCRETE = 1

class HMM:
    ''' Code for a hidden Markov Model '''

    def __init__(self, states, features, contOrDisc, numVals):
        ''' Initialize the HMM.
            Input:
                states: a list of the hidden state possible values
                features: a list of feature names
                contOrDisc: a dictionary mapping feature names to integers
                    representing whether the feature is continuous or discrete
                numVals: a dictionary mapping names of discrete features to
                    the number of values that feature can take on. '''
        self.states = states 
        self.isTrained = False
        self.featureNames = features
        self.featuresCorD = contOrDisc
        self.numVals = numVals

        # All the probabilities start uninitialized until training
        self.priors = None
        self.emissions = None   #evidence model
        self.transitions = None #transition model

    def train(self, trainingData, trainingLabels):
        ''' Train the HMM on the fully observed data using MLE '''
        print "Training the HMM... "
        self.isTrained = True
        self.trainPriors( trainingData, trainingLabels )
        self.trainTransitions( trainingData, trainingLabels )
        self.trainEmissions( trainingData, trainingLabels ) 
        print "HMM trained"
        print "Prior probabilities are:", self.priors
        print "Transition model is:", self.transitions
        print "Evidence model is:", self.emissions

    def trainPriors( self, trainingData, trainingLabels ):
        ''' Train the priors based on the data and labels '''
        # Set the prior probabilities
        priorCounts = {}
        for s in self.states:
            priorCounts[s] = 0
        for labels in trainingLabels:
            priorCounts[labels[0]] += 1

        self.priors = {}
        for s in self.states:
            self.priors[s] = max(1.0,float(priorCounts[s]))/len(trainingLabels)
        

    def trainTransitions( self, trainingData, trainingLabels ):
        ''' Give training data and labels, train the transition model '''
        # Set the transition probabilities
        # First initialize the transition counts
        transitionCounts = {}
        for s in self.states:
            transitionCounts[s] = {}
            for s2 in self.states:
                transitionCounts[s][s2] = 0
                
        for labels in trainingLabels:
            if len(labels) > 1:
                lab1 = labels[0]
                for lab2 in labels[1:]:
                    transitionCounts[lab1][lab2] += 1
                    lab1 = lab2
                    
        self.transitions = {}
        for s in transitionCounts.keys():
            self.transitions[s] = {}
            totForS = sum(transitionCounts[s].values())
            for s2 in transitionCounts[s].keys():
                self.transitions[s][s2] = max(1.0,float(transitionCounts[s][s2]))/float(totForS)


    def trainEmissions( self, trainingData, trainingLabels ):
        ''' given training data and labels, train the evidence model.  '''
        self.emissions = {}
        featureVals = {}
        for s in self.states:
            self.emissions[s] = {}
            featureVals[s] = {}
            for f in self.featureNames:
                # there might be no instance in continuous case
                featureVals[s][f] = []

        # Now gather the features for each state
        for i in range(len(trainingData)):
            oneSketchFeatures = trainingData[i]
            oneSketchLabels = trainingLabels[i]
            
            for j in range(len(oneSketchFeatures)):
                features = oneSketchFeatures[j]
                for f in features.keys():
                    featureVals[oneSketchLabels[j]][f].append(features[f])

        # Do a slightly different thing for conituous vs. discrete features
        for s in featureVals.keys():
            for f in featureVals[s].keys():
                if self.featuresCorD[f] == CONTINUOUS:
                    # Use a gaussian representation, so just find the mean and standard dev of the data
                    # mean is just the sample mean
                    mean = sum(featureVals[s][f])/len(featureVals[s][f])
                    sigmasq = sum([(x - mean)**2 for x in featureVals[s][f]]) / len(featureVals[s][f])
                    sigma = math.sqrt(sigmasq)
                    self.emissions[s][f] = [mean, sigma]
                if self.featuresCorD[f] == DISCRETE:
                    # If the feature is discrete then the CPD is a list
                    # We assume that feature values are integer, starting
                    # at 0.  This assumption could be generalized.
                    counter = 0
                    self.emissions[s][f] = [1]*self.numVals[f]  # Use add 1 smoothing
                    for fval in featureVals[s][f]:
                        self.emissions[s][f][fval] += 1
                    # Now we have counts of each feature and we need to normalize
                    for i in range(len(self.emissions[s][f])):
                        self.emissions[s][f][i] = max(1.0,self.emissions[s][f][i]) / float(len(featureVals[s][f])+self.numVals[f])

              
    def label( self, data ):
        ''' Find the most likely labels for the sequence of data
            This is an implementation of the Viterbi algorithm  '''
        # You will implement this function
        #list of 'text' and 'drawing'
        labelList = []
        transitionList = []
        prevPartialProb = {}
        finalState = ''
        # now have a sequence of length [{'length': 1}, {'length': 1}, {'length': 1}, {'length': 1}, {'length': 1}, {'length': 1}, {'length': 1}, {'length': 1}, {'length': 0}]
        #self.priors {'text': 0.5272727272727272, 'drawing': 0.4727272727272727}
        #self.transitions {'text': {'text': 0.8217821782178217, 'drawing': 0.1782178217821782}, 'drawing': {'text': 0.10454908220271349, 'drawing': 0.8954509177972865}}
        #self.emissions {'text': {'length': [0.6366083445491252, 0.3633916554508748]}, 'drawing': {'length': [0.3166144200626959, 0.6833855799373041]}}

        for fIndex in range(len(data)):
            feature = data[fIndex]
            tempPartialProb = {}
            tempPrevState = {} 
            for prior in self.priors:
                emissionProb = math.log(self.getEmissionProb(prior, feature),2)
                # 1st state calculation 
                if fIndex == 0:
                    tempPartialProb[prior] = math.log(self.priors[prior],2) + emissionProb
                # other state calculation
                else:
                    tempPartialProb[prior] = float('-inf');
                    for prevState in self.transitions:
                        tempProb = prevPartialProb[prevState] + math.log(self.transitions[prevState][prior],2) + emissionProb
                        if tempProb > tempPartialProb[prior]:
                            tempPartialProb[prior] = tempProb
                            tempPrevState[prior] = prevState

            #update previous state partial prob
            prevPartialProb = tempPartialProb
            #update transition list
            if fIndex != 0:
                transitionList.append(tempPrevState)
        finalState = max(prevPartialProb.iteritems(), key=operator.itemgetter(1))[0]

        labelList.append(finalState)
        #fill labelList from the final state 
        # print transitionList
        for prev in reversed(transitionList):
            labelList.insert(0,prev[finalState])
            finalState = prev[finalState]
        print "final state partial probability: " + str(prevPartialProb)
        # print "label sequence generated: " + str(labelList)
        #return a list of labels
        return labelList
    
    def getEmissionProb( self, state, features ):
        ''' Get P(features|state).
            Consider each feature independent so
            P(features|state) = P(f1|state)*P(f2|state)*...*P(fn|state). '''
        prob = 1.0
        for f in features:
            if self.featuresCorD[f] == CONTINUOUS:
                # calculate the gaussian prob
                fval = features[f]
                mean = self.emissions[state][f][0]
                sigma = self.emissions[state][f][1]
                g = math.exp((-1*(fval-mean)**2) / (2*sigma**2))
                g = g / (sigma * math.sqrt(2*math.pi))
                prob *= g
            if self.featuresCorD[f] == DISCRETE:
                fval = features[f]
                prob *= self.emissions[state][f][fval]
                
        return prob
        


class StrokeLabeler:
    def __init__(self):
        ''' Inialize a stroke labeler. '''
        self.labels = ['text', 'drawing']
        # a map from labels in files to labels we use here
        drawingLabels = ['Wire', 'AND', 'OR', 'XOR', 'NAND', 'NOT']
        textLabels = ['Label']
        self.labels = ['drawing', 'text']
        
        self.labelDict = {}
        for l in drawingLabels:
            self.labelDict[l] = 'drawing'
        for l in textLabels:
            self.labelDict[l] = 'text'

        # Define the features to be used in the featurefy function
        # if you change the featurefy function, you must also change
        # these data structures.
        # featureNames is just a list of all features.
        # contOrDisc is a dictionary mapping each feature
        #    name to whether it is continuous or discrete
        # numFVals is a dictionary specifying the number of legal values for
        #    each discrete feature
        self.featureNames = ['length','sumOfCurvature','ratioOfWidthHeight','toSide','timeDuration']
        self.contOrDisc = {}
        self.numFVals = {}
        self.featureIntervals = {}
        for featureName in self.featureNames:
            self.contOrDisc[featureName] = DISCRETE
            self.numFVals[featureName] = 2
        
    def featurefy( self, strokes):
        ''' Converts the list of strokes into a list of feature dictionaries
            suitable for the HMM
            The names of features used here have to match the names
            passed into the HMM'''
        ret = []
        for s in strokes:
            d = {}  # The feature dictionary to be returned for one stroke

            # If we wanted to use length as a continuous feature, we
            # would simply use the following line to set its value
            # d['length'] = s.length()

            # To use it as a discrete feature, we have to "bin" it, that is
            # we define ranges for "short" (<300 units) and "long" (>=300 units)
            # Short strokes get a discrete value 0, long strokes get
            # discrete value 1.
            # Note that these bins were determined by trial and error, and my
            # looking at the length data, to determine what a good discriminating
            # cutoff would be.  You might choose to add more bins
            # or to change the thresholds.  For any other discrete feature you
            # add, it's up to you to determine how many and what bins you want
            # to use.  This is an important process and can be tricky.  Try
            # to use a principled approach (i.e., look at the data) rather
            # than just guessing.
            '''
            With this two line of codes, we import all features we want to use from the self.featureNames list
            Then we use the already-calculated values to classify continous features, which also means that we don't have any discrete values to give to HMM model
            The calculation of those classification values are in 'generateFeatureIntervals' function
            '''
            for featureName,featureInterval in self.featureIntervals.items():
                d[featureName] = 1 if s.featureValues[featureName] > featureInterval else 0
                
            # We can add more features here just by adding them to the dictionary
            # d as we did with length.  Remember that when you add features,
            # you also need to add them to the three member data structures
            # above in the contructor: self.featureNames, self.contOrDisc,
            #    self.numFVals (for discrete features only)


            ret.append(d)  # append the feature dictionary to the list
            
        return ret
    
    def generateFeatureIntervals(self,allStrokes,allLabels,intervalNums = 10):
        result = {}
        for featureName in self.featureNames:
            result[featureName] = {'text':[],'drawing':[]}
        for i in range(len(allStrokes)):
            for j in range(len(allStrokes[i])):
                for featureName in self.featureNames:
                    result[featureName][allLabels[i][j]].append(allStrokes[i][j].featureValues[featureName])
        # self.result =  result
        for featureName in self.featureNames:
            textList = result[featureName]['text']
            drawingList = result[featureName]['drawing']
            averText = 0 if len(textList) == 0 else float(sum(textList))/len(textList)
            averDrawing = 0 if len(drawingList) == 0 else float(sum(drawingList))/len(drawingList)
            target = (averText,2)
            step = float(averDrawing-averText)/intervalNums
            dPoint = averText
            for i in range(1,intervalNums): 
                dPoint += step
                list1 = ['drawing' for v in drawingList if v>dPoint] + ['text' for v in textList if v>dPoint]
                list2 = ['drawing' for v in drawingList if v<=dPoint] + ['text' for v in textList if v<=dPoint]
                entropy = self.calculateEntropy(list1,list2)
                if entropy<target[1]:
                    target = (dPoint,entropy)
            self.featureIntervals[featureName] = target[0]




    def calculateEntropy(self, list1, list2):
        prob1 = float(len(list1))/(len(list1) + len(list2))
        prob2 = float(len(list2))/(len(list1) + len(list2))  
        numText1 = 0
        numText2 = 0        
        for l1 in list1:
            if l1 == 'text':
                numText1 += 1
        probText1 = float(numText1)/len(list1)
        probDrawing1 = 1 - probText1
        for l2 in list2:
            if l2 == 'text':
                numText2 += 1
        probText2 = float(numText2)/len(list2)
        probDrawing2 = 1 - probText2
        list1Entropy = 0 if probText1 * probDrawing1 == 0 else -probText1*math.log(probText1,2) - probDrawing1*math.log(probDrawing1,2)
        list2Entropy = 0 if probText2 * probDrawing2 == 0 else -probText2*math.log(probText2,2) - probDrawing2*math.log(probDrawing2,2)
        conditionEntropy = prob1 * list1Entropy + prob2 * list2Entropy

    def trainHMM( self, trainingFiles ):
        ''' Train the HMM '''
        self.hmm = HMM( self.labels, self.featureNames, self.contOrDisc, self.numFVals )
        allStrokes = []
        allLabels = []
        for f in trainingFiles:
            print "Loading file", f, "for training"
            strokes, labels = self.loadLabeledFile( f )
            allStrokes.append(strokes)
            allLabels.append(labels)
        self.allStrokes = allStrokes
        self.allLabels = allLabels
        self.generateFeatureIntervals(allStrokes,allLabels)
        allObservations = [self.featurefy(s) for s in allStrokes]
        print "original labels:" + str(labels)
        self.hmm.train(allObservations, allLabels)

    def trainHMMDir( self, trainingDir ):
        ''' train the HMM on all the files in a training directory '''

        for fFileObj in os.walk(trainingDir):
            lFileList = fFileObj[2]
            break
        goodList = []
        for x in lFileList:
            if not x.startswith('.'):
                goodList.append(x)
        
        tFiles = [ trainingDir + "/" + f for f in goodList ] 
        self.trainHMM(tFiles)

    def featureTest( self, strokeFile ):
        ''' Loads a stroke file and tests the feature functions '''
        strokes, labels = self.loadLabeledFile( strokeFile )
        for i in range(len(strokes)):
            print " "
            print strokes[i].substrokeIds[0]
            print "Label is", labels[i]
            print "Length is", strokes[i].length()
            print "Curvature is", strokes[i].sumOfCurvature(abs)
    
    def labelFile( self, strokeFile, outFile ):
        ''' Label the strokes in the file strokeFile and save the labels
            (with the strokes) in the outFile '''
        print "Labeling file", strokeFile
        strokes = self.loadStrokeFile( strokeFile )
        labels = self.labelStrokes( strokes )
        print "Labeling done, saving file as", outFile
        self.saveFile( strokes, labels, strokeFile, outFile )

    def labelStrokes( self, strokes ):
        ''' return a list of labels for the given list of strokes '''
        if self.hmm == None:
            print "HMM must be trained first"
            return []
        strokeFeatures = self.featurefy(strokes)
        # print strokeFeatures
        return self.hmm.label(strokeFeatures)

    def saveFile( self, strokes, labels, originalFile, outFile ):
        ''' Save the labels of the stroke objects and the stroke objects themselves
            in an XML format that can be visualized by the labeler.
            Need to input the original file from which the strokes were read
            so that we can retrieve a lot of data that we don't store here'''
        sketch = xml.dom.minidom.parse(originalFile)
        # copy most of the data, including all points, substrokes, strokes
        # then just add the shapes onto the end
        impl =  xml.dom.minidom.getDOMImplementation()
        
        newdoc = impl.createDocument(sketch.namespaceURI, "sketch", sketch.doctype)
        top_element = newdoc.documentElement

        # Add the attibutes from the sketch document
        for attrib in sketch.documentElement.attributes.keys():
            top_element.setAttribute(attrib, sketch.documentElement.getAttribute(attrib))

        # Now add all the children from sketch as long as they are points, strokes
        # or substrokes
        sketchElem = sketch.getElementsByTagName("sketch")[0]
        for child in sketchElem.childNodes:
            if child.nodeType == xml.dom.Node.ELEMENT_NODE:
                if child.tagName == "point":
                    top_element.appendChild(child)
                elif child.tagName == "shape":
                    if child.getAttribute("type") == "substroke" or \
                       child.getAttribute("type") == "stroke":
                        top_element.appendChild(child)    

        # Finally, add the new elements for the labels
        for i in range(len(strokes)):
            # make a new element
            newElem = newdoc.createElement("shape")
            # Required attributes are type, name, id and time
            newElem.setAttribute("type", labels[i])
            newElem.setAttribute("name", "shape")
            newElem.setAttribute("id", guid.generate() )
            newElem.setAttribute("time", str(strokes[i].points[-1][2]))  # time is finish time

            # Now add the children
            for ss in strokes[i].substrokeIds:
                ssElem = newdoc.createElement("arg")
                ssElem.setAttribute("type", "substroke")
                ssElem.appendChild(newdoc.createTextNode(ss))
                newElem.appendChild(ssElem)
                
            top_element.appendChild(newElem)
            

        # Write to the file
        filehandle = open(outFile, "w")
        newdoc.writexml(filehandle)
        filehandle.close()

        # unlink the docs
        newdoc.unlink()
        sketch.unlink()

    def loadStrokeFile( self, filename ):
        ''' Read in a file containing strokes and return a list of stroke
            objects '''
        sketch = xml.dom.minidom.parse(filename)
        # get the points
        points = sketch.getElementsByTagName("point")
        pointsDict = self.buildDict(points)
    
        # now get the strokes by first getting all shapes
        allShapes = sketch.getElementsByTagName("shape")
        shapesDict = self.buildDict(allShapes)

        strokes = []
        for shape in allShapes:
            if shape.getAttribute("type") == "stroke":
                strokes.append(self.buildStroke( shape, shapesDict, pointsDict ))

        # I THINK the strokes will be loaded in order, but make sure
        if not self.verifyStrokeOrder(strokes):
            print "WARNING: Strokes out of order"

        sketch.unlink()
        return strokes

    def verifyStrokeOrder( self, strokes ):
        ''' returns True if all of the strokes are temporally ordered,
            False otherwise. '''
        time = 0
        ret = True
        for s in strokes:
            if s.points[0][2] < time:
                ret = False
                break
            time = s.points[0][2]
        return ret

    def buildDict( self, nodesWithIdAttrs ):
        ret = {}
        for n in nodesWithIdAttrs:
            idAttr = n.getAttribute("id")
            ret[idAttr] = n
        
        return ret

    def buildStroke( self, shape, shapesDict, pointDict ):
        ''' build and return a stroke object by finding the substrokes and points
            in the shape object '''
        ret = Stroke( shape.getAttribute("id") )
        points = []
        # Get the children of the stroke
        last = None
        for ss in shape.childNodes:
            if ss.nodeType != xml.dom.Node.ELEMENT_NODE \
               or ss.getAttribute("type") != "substroke":
                continue

            # Add the substroke id to the stroke object
            ret.addSubstroke(ss.firstChild.data)
            
            # Find the shape with the id of this substroke
            ssShape = shapesDict[ss.firstChild.data]

            # now get all the points associated with this substroke
            # We'll filter points that don't move here
            for ptObj in ssShape.childNodes:
                if ptObj.nodeType != xml.dom.Node.ELEMENT_NODE \
                   or ptObj.getAttribute("type") != "point":
                    continue
                pt = pointDict[ptObj.firstChild.data]
                x = int(pt.getAttribute("x"))
                y = int(pt.getAttribute("y"))
                time = int(pt.getAttribute("time"))
                if last == None or last[0] != x or last[1] != y:  # at least x or y is different
                    points.append((x, y, time))
                    last = (x, y, time)
        ret.setPoints(points)
        ret.featureValues['length'] = ret.length()
        ret.featureValues['sumOfCurvature'] = ret.sumOfCurvature()
        ret.featureValues['ratioOfWidthHeight'] = ret.ratioOfWidthHeight()
        ret.featureValues['timeDuration'] = ret.timeDuration()
        return ret
                

    def loadLabeledFile( self, filename ):
        ''' load the strokes and the labels for the strokes from a labeled file.
            return the strokes and the labels as a tuple (strokes, labels) '''
        sketch = xml.dom.minidom.parse(filename)
        # get the points
        points = sketch.getElementsByTagName("point")
        pointsDict = self.buildDict(points)
    
        # now get the strokes by first getting all shapes
        allShapes = sketch.getElementsByTagName("shape")
        shapesDict = self.buildDict(allShapes)

        strokes = []
        substrokeIdDict = {}
        left,right = float('inf'),float('-inf')
        for shape in allShapes:
            if shape.getAttribute("type") == "stroke":
                stroke = self.buildStroke( shape, shapesDict, pointsDict )
                left,right = min(stroke.minX,left),max(stroke.maxX,right)
                strokes.append(self.buildStroke( shape, shapesDict, pointsDict ))
                substrokeIdDict[stroke.strokeId] = stroke
            else:
                # If it's a shape, then just store the label on the substrokes
                for child in shape.childNodes:
                    if child.nodeType != xml.dom.Node.ELEMENT_NODE \
                       or child.getAttribute("type") != "substroke":
                        continue
                    substrokeIdDict[child.firstChild.data] = shape.getAttribute("type")
        for stroke in strokes:
            stroke.featureValues['toSide'] = stroke.toSide(left,right)

        # I THINK the strokes will be loaded in order, but make sure
        if not self.verifyStrokeOrder(strokes):
            print "WARNING: Strokes out of order"

        # Now put labels on the strokes
        labels = []
        noLabels = []
        for stroke in strokes:
            # Just give the stroke the label of the first substroke in the stroke
            ssid = stroke.substrokeIds[0]
            if not self.labelDict.has_key(substrokeIdDict[ssid]):
                # If there is no label, flag the stroke for removal
                noLabels.append(stroke)
            else:
                labels.append(self.labelDict[substrokeIdDict[ssid]])

        for stroke in noLabels:
            strokes.remove(stroke)
            
        sketch.unlink()
        if len(strokes) != len(labels):
            print "PROBLEM: number of strokes and labels must match"
            print "numStrokes is", len(strokes), "numLabels is", len(labels)
        return strokes, labels

    def confusion(self,trueLabels, classifications):
        result = {'drawing':{'drawing':0,'text':0},'text':{'drawing':0,'text':0}}
        for i in range(len(trueLabels)):
            result[trueLabels[i]][classifications[i]] += 1    
        print "confusion table: " + str(result)
        print "accuracy: " + str(float(result['drawing']['drawing']+result['text']['text'])/(result['drawing']['drawing']+result['text']['text']+result['text']['drawing']+result['drawing']['text']))
        return result

    def validateAll(self):
        self.classifications = self.labelStrokes(flatten(self.allStrokes))
        self.confusion(flatten(self.allLabels),self.classifications)


class Stroke:
    ''' A class to represent a stroke (series of xyt points).
        This class also has various functions for computing stroke features. '''
    def __init__(self, strokeId):
        self.strokeId = strokeId
        self.substrokeIds = []   # Keep around the substroke ids for writing back to file
        self.featureValues = {}
        
    def __repr__(self):
        ''' Return a string representation of the stroke '''
        return "[Stroke " + self.strokeId + "]"

    def addSubstroke( self, substrokeId ):
        ''' Add a substroke Id to the stroke '''
        self.substrokeIds.append(substrokeId)

    def setPoints( self, points ):
        ''' Set the points for the stroke '''
        self.points = points


    # Feature functions follow this line
    def length( self ):
        ''' Returns the length of the stroke '''
        ret = 0
        prev = self.points[0]
        for p in self.points[1:]:
            # use Euclidean distance
            xdiff = p[0] - prev[0]
            ydiff = p[1] - prev[1]
            ret += math.sqrt(xdiff**2 + ydiff**2)
            prev = p
        return ret

    def ratioOfWidthHeight(self):
        '''this is the ratio of stroke boundary's width to height'''
        self.minX,self.minY,self.maxX,self.maxY = float('inf'),float('inf'),float('-inf'),float('-inf')
        for p in self.points:
            self.minX,self.minY,self.maxX,self.maxY = min(self.minX,float(p[0])),min(self.minY,float(p[1])),max(self.maxX,float(p[0])),max(self.maxY,float(p[1]))
        bWidth,bHeight = self.maxX-self.minX,self.maxY-self.minY
        return 0 if bWidth*bHeight == 0 else min(bWidth/bHeight,bHeight/bWidth)

    def toSide(self,left,right):
        return min(right-self.maxX,self.minX-left)

    def timeDuration(self):
        minT,maxT = float('inf'),float('-inf')
        for p in self.points:
            minT,maxT = min(minT,float(p[2])),max(maxT,float(p[2]))
        return maxT-minT

    def sumOfCurvature(self, func=lambda x: x, skip=1):
        ''' Return the normalized sum of curvature for a stroke.
            func is a function to apply to the curvature before summing
                e.g., to find the sum of absolute value of curvature,
                you could pass in abs
            skip is a smoothing constant (how many points to skip)
        '''
        if len(self.points) < 2*skip+1:
            return 0
        ret = 0
        second = self.points[0]
        third = self.points[1*skip]
        for p in self.points[2*skip::skip]:
            
            first = second
            second = third
            third = p
            ax = second[0] - first[0]
            ay = second[1] - first[1]
            bx = third[0] - second[0]
            by = third[1] - second[1]
            
            lena = math.sqrt(ax**2 + ay**2)
            lenb = math.sqrt(bx**2 + by**2)

            dotab = ax*bx + ay*by
            arg = float(dotab)/float(lena*lenb)

            # Fix floating point precision errors
            if arg > 1.0:
                arg = 1.0
            if arg < -1.0:
                arg = -1.0

            curv = math.acos(arg)

            # now we have to find the sign of the curvature
            # get the angle betwee the first vector and the x axis
            anga = math.atan2(ay, ax)
            # and the second
            angb = math.atan2(by, bx)
            # now compare them to get the sign.
            if not(angb < anga and angb > anga-math.pi):
                curv *= -1
            ret += func(curv)

        return ret / len(self.points)

    # You can (and should) define more features here


