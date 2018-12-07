

Data file 20181115-sealSpotter_classifications_anonymised.csv was created by Ross Holmberg at Phillip Island Nature Parks, for the purpose of sharing data as part of a collaboration between Phillip Island Nature Parks and Monash University. Classification data have been collected using Phillip Island Nature Parks' "SealSpotter" web portal, accepting inputs from both experts in the field, and volunteers from around the world.

For correspondence relating to this dataset, please contact either:
    Dr Rebecca McIntosh - rmcintosh@penguins.org.au
    Ross Holmberg - rholmberg@penguins.org.au

This csv file, and the associated image files, are being shared as per the agreement between Phillip Island Nature Parks and Monash University relevant to this project (created over time both in conversation and in writing). As per that agreement, the data is not to be shared or used in any way outside of this project without express permission from Phillip Island Nature Parks.

User information has been anonymised, other than a classification between "expert" and "non-expert" users.

Timestamps are formatted using the ISO8601 standard ( "YYYY-MM-DD hh:mm:ss" )

#### Site specific notes:

    Surveys are generally captured at 40m Above Sea Level (ASL), however the distance between the camera and the ground can vary significantly based on the topography of the site. For example, Seal Rocks has a ~ 15m ASL plateau, and Rag Island is made up of steep slopes near the coasts, and a high plateau (~ 35m ASL at a guess) in the middle.
    Deen Maar is particularly problematic here, to the extent that it may be best to disregard the images from there completely due to low quality and inconsistent subject distances.
    
    The Skerries was captured at a slightly lower altitude (35m ASL) due to the small site area and flatter topography. This results in slightly better image resolution, and so will create a slight difference in the pixel dimensions of the animals.



#### Columns:

	 `timestamp_server` - the moment the data row was written to the database. Should be the same (or within 1s) for each submission of data (ie: each image classification per user)

	 `timestamp_client` - the moment the mouse was clicked on the point. Can be different for the same submission of data

	 `imagefile` - file name of the image classified.

	 `user_anonymised` - integer value randomly assigned per user. Each integer refers to a specific user name, which will remain consistent between user sessions, as long as the user signs in to the portal under the same name.

	 `expert` - TRUE where the user is considered an expert in the field, and whose classifications should be trusted more highly than those of non-experts.

	 `type` - seal type associated with the point.
	    In all cases, the `type` classes available to the user as points are:
            "adult_juv" - adults (both Male and Female), and juveniles (all young except for the most recent cohort)
            "pup" - the most recent cohort, here less than ~ 2 months old
            "entangled" - an animal visibly entangled in marine debris
	 	"zerocount" is used here to ensure that a row is included in the dataset marking an image as having no seals.
	 	"comment" is used to separate a user's comment from point data.
	 	both "zerocount" and "comment" are given point coordinates 0,0
        Note that when an animal is classified as "entangled", it most likely will NOT have a separate age class label.

	 `pointsImageX` & `pointsImageY` - the pixel coordinates of the point within the image
	 	0,0 is the bottom left corner of an image

	 `imgWidth` & `imgHeight` - the total size of the image, in pixels

	 `comments` - user entered comment strings

	 `survey_loc` - the site at which the survey was conducted
	 
	 `survey_date` - the date on which the survey was conducted
	 
	 
	 
