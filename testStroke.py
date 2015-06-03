execfile("StrokeHMM.py")
sl = StrokeLabeler()
sl.trainHMMDir("trainingFiles/")
sl.labelFile( "trainingFiles/0128_1.6.1.labeled.xml", "out.xml")
