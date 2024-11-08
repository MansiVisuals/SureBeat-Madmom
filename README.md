# SureBeat-Madmom

Here’s a step-by-step guide to manually setting up SureBeat on macOS.
This guide will walk you through installing all required dependencies and configuring SureBeat within DaVinci Resolve.

1. Install Dependencies

SureBeat relies on several key dependencies, including Python, FFmpeg, and madmom. Make sure you follow each step to ensure all dependencies are properly installed.

1.1 Install Python

	1.	Check Python Version: SureBeat is tested with Python 3.10 and 3.13, so ensure you have one of these versions installed.

python3 --version


	2.	Install Python (if needed):
	•	Download from the official Python website, or install with Homebrew:

https://www.python.org/downloads/

brew install python


1.2 Install FFmpeg

	1.	Check FFmpeg: Verify if FFmpeg is installed.

ffmpeg -version


	2.	Install FFmpeg (if needed):
 
https://ffmpeg.org/download.html download binarie from here and place into /usr/local/bin

brew install ffmpeg


1.3 Install Python Packages and madmom

SureBeat requires the madmom library, along with additional libraries like numpy, scipy, and mido.
	1.	Install pip packages:

pip3 install numpy scipy mido

	2.	Install madmom:
	•	Install the specific version (0.17) of madmom from the GitHub source:

pip3 install git+https://github.com/CPJKU/madmom.git@main


2. Download SureBeat Files

	1.	Download SureBeat Files:
	•	Download SureBeat.lua and beat_detection.py from the SureBeat repository on GitHub:
	•	SureBeat.lua
	•	beat_detection.py
	2.	Create DaVinci Resolve Directory:
	•	DaVinci Resolve expects these files in a specific folder. Create the following directory if it does not already exist:

mkdir -p "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility/SureBeat"


	3.	Place Files in the Directory:
	•	Move or copy SureBeat.lua and beat_detection.py to the directory:

cp path/to/SureBeat.lua "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility/SureBeat/"
cp path/to/beat_detection.py "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility/SureBeat/"

3. Verify Setup

To confirm that SureBeat is properly set up, you can perform the following checks:
	1.	Check Python Installation:

python3 -c "import madmom; print(madmom.__version__)"

	•	This should print 0.17, verifying that madmom is correctly installed.

	2.	Check DaVinci Resolve Script Files:
	•	Open Finder and navigate to:

/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility/SureBeat/


	•	Ensure both SureBeat.lua and beat_detection.py are present.

	3.	Test the Python Script:
	•	Run the beat_detection.py script from the terminal to check if it works independently:

python3 "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility/SureBeat/beat_detection.py" path/to/your_audio_file.mp3

	•	Replace path/to/your_audio_file.mp3 with an actual audio file path. This command should output beat and tempo information if everything is set up correctly.
 

4. Using SureBeat in DaVinci Resolve

	1.	Open DaVinci Resolve.
	2.	Navigate to the Fusion Tab.
	3.	Access the SureBeat Tool:
	•	In the Fusion tab, open the scripts menu and select SureBeat to run it.

Troubleshooting

If you encounter issues, here are a few troubleshooting steps:
	•	Python Compatibility: Ensure that the Python version used matches the one installed (check with python3 --version).
	•	Permissions: If you get permissions errors, ensure you have admin rights for placing files in /Library/Application Support/Blackmagic Design.
	•	Dependency Check: Verify that all required packages are installed by running:

pip3 show numpy scipy mido madmom

By following these steps, SureBeat should be installed and fully functional in DaVinci Resolve on macOS!