#!/bin/sh

show_menu(){
  NORMAL=`echo "\033[m"`
  MENU=`echo "\033[36m"`
  NUMBER=`echo "\033[33m"`
  FGRED=`echo "\033[41m"`
  RED_TEXT=`echo "\033[31m"`
  ENTER_LINE=`echo "\033[33m"`
  echo -e "${MENU}*********************************************${NORMAL}"
  echo -e "${MENU}**${NUMBER} 1)${MENU} CAL500 ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 2)${MENU} FMA ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 3)${MENU} MagnaTagATune ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 4)${MENU} MillionSong ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 5)${MENU} Music2All ${NORMAL}"
  echo -e "${MENU}*********************************************${NORMAL}"
  echo -e "${ENTER_LINE}Please enter a menu option and enter or ${RED_TEXT} enter to exit. ${NORMAL}"
  read opt

  while [ opt != '' ]
  do
    if [[ $opt = "" ]]; then
      exit;
    else
      case $opt in
      1) clear;
      export database_name="CAL500"
      option_picked $database_name;    
      sub_menu;
      ;;

      2) clear;
      export database_name="FMA"
      option_picked $database_name;    
      sub_menu;
      ;;

      3) clear;
      export database_name="MagnaTagATune"
      option_picked $database_name;    
      sub_menu;
      ;;

      4) clear;
      export database_name="MillionSong"
      option_picked $database_name;    
      sub_menu;
      ;;

      5) clear;
      export database_name="Music2All"
      option_picked $database_name;    
      sub_menu;
      ;;

      x)exit;
      ;;

      \n)exit;
      ;;

      *)clear;
      option_picked "Pick an option from the menu";
      show_menu;
      ;;
      esac
    fi
  done
}

function option_picked() {
  COLOR='\033[01;31m'
  RESET='\033[00;00m'
  MESSAGE=${@:-"${RESET}Error: No message passed"}
  echo -e "${COLOR}${MESSAGE}${RESET}"
}

pyclean () {
  find . -regex '^.*\(__pycache__\|\.py[co]\)$' -delete
}

sub_menu(){
  NORMAL=`echo "\033[m"`
  MENU=`echo "\033[36m"`
  NUMBER=`echo "\033[33m"`
  FGRED=`echo "\033[41m"`
  RED_TEXT=`echo "\033[31m"`
  ENTER_LINE=`echo "\033[33m"`
  echo -e "${MENU}*********************************************${NORMAL}"
  echo -e "${MENU}**${RED_TEXT} 0)${MENU} Return to Menu ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 1)${MENU} Generate Structure ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 2)${MENU} Generate One Hot Vector, Samples per Time, Cardinality and Density ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 3)${MENU} Check Database ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 4)${MENU} Generate Spectrogram ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 5)${MENU} Generate Autoencoders ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 6)${MENU} Generate Train/Test/Validation ${NORMAL}"
  echo -e "${MENU}*********************************************${NORMAL}"
  echo -e "${MENU}**${NUMBER} 10)${MENU} Model 1 ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 11)${MENU} Model 2 ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 12)${MENU} Model 3 ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 13)${MENU} Model 4 ${NORMAL}"
  echo -e "${MENU}*********************************************${NORMAL}"
  echo -e "${MENU}**${NUMBER} 20)${MENU} Model 5 ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 21)${MENU} Model 6 ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 22)${MENU} Model 7 ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 23)${MENU} Model 8 ${NORMAL}"
  echo -e "${MENU}*********************************************${NORMAL}"
  echo -e "${MENU}**${NUMBER} 30)${MENU} Model 9 ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 31)${MENU} Model 10 ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 32)${MENU} Model 11 ${NORMAL}"
  echo -e "${MENU}**${NUMBER} 33)${MENU} Model 12 ${NORMAL}"
  echo -e "${MENU}*********************************************${NORMAL}"
  echo -e "${ENTER_LINE}Please enter a menu option and enter or ${RED_TEXT}enter to exit. ${NORMAL}"
  read sub1
  while [ sub1 != '' ]
  do
    if [[ $sub1 = "" ]]; then
      exit;
    else
      case $sub1 in
      0) clear;
      pyclean;
      option_picked "Return to Menu";
      show_menu;
      ;;

      1) clear;
      pyclean;
      option_picked "Generate Structure";
      python3 $(pwd)/src/generate_structure.py
      sub_menu;
      ;;

      2) clear;
      pyclean;
      option_picked "Generate One Hot Vector, Samples per Time, Cardinality and Density";
      python3 $(pwd)/src/generate_info_data.py
      sub_menu;
      ;;

      3) clear;
      pyclean;
      option_picked "Check Data";
      python3 $(pwd)/src/check_data.py
      sub_menu;
      ;;

      4) clear;
      pyclean;
      option_picked "Generate Spectrogram";
      python3 $(pwd)/src/generate_spectrogram.py
      sub_menu;
      ;;

      5) clear;
      pyclean;
      option_picked "Generate Autoencoders Chromagram";
      python3 $(pwd)/src/generate_autoencoders_chromagram.py
      option_picked "Generate Autoencoders Mel Spectrogram";
      python3 $(pwd)/src/generate_autoencoders_mel_spectrogram.py
      option_picked "Generate Autoencoders MFCC";
      python3 $(pwd)/src/generate_autoencoders_mfcc.py
      option_picked "Generate Autoencoders STFT";
      python3 $(pwd)/src/generate_autoencoders_stft.py
      sub_menu;
      ;;
  
      6) clear;
      pyclean;
      option_picked "Generate TRAIN/TEST/VALIDATION";
      python3 $(pwd)/src/generate_holdout.py      
      sub_menu;
      ;;

      10) clear;
      pyclean;
      option_picked "Model 1 - First Stage";
      python3 $(pwd)/database/$database_name/src/model-1/first_stage.py
      option_picked "Model 1 - Second Stage";
      python3 $(pwd)/database/$database_name/src/model-1/second_stage.py
      sub_menu;
      ;;

      11) clear;
      pyclean;
      option_picked "Model 2 - First Stage";
      python3 $(pwd)/database/$database_name/src/model-2/first_stage.py
      option_picked "Model 2 - Second Stage";
      python3 $(pwd)/database/$database_name/src/model-2/second_stage.py
      sub_menu;
      ;;

      12) clear;
      pyclean;
      option_picked "Model 3 - First Stage";
      python3 $(pwd)/database/$database_name/src/model-3/first_stage.py
      option_picked "Model 3 - Second Stage";
      python3 $(pwd)/database/$database_name/src/model-3/second_stage.py
      sub_menu;
      ;;


      13) clear;
      pyclean;
      option_picked "Model 4 - First Stage";
      python3 $(pwd)/database/$database_name/src/model-4/first_stage.py
      option_picked "Model 4 - Second Stage";
      python3 $(pwd)/database/$database_name/src/model-4/second_stage.py
      sub_menu;
      ;;

      20) clear;
      pyclean;
      option_picked "Model 5 - First Stage";
      python3 $(pwd)/database/$database_name/src/model-5/first_stage.py
      option_picked "Model 5 - Second Stage";
      python3 $(pwd)/database/$database_name/src/model-5/second_stage.py
      sub_menu;
      ;;

      21) clear;
      pyclean;
      option_picked "Model 6 - First Stage";
      python3 $(pwd)/database/$database_name/src/model-6/first_stage.py
      option_picked "Model 6 - Second Stage";
      python3 $(pwd)/database/$database_name/src/model-6/second_stage.py
      sub_menu;
      ;;

      22) clear;
      pyclean;
      option_picked "Model 7 - First Stage";
      python3 $(pwd)/database/$database_name/src/model-7/first_stage.py
      option_picked "Model 7 - Second Stage";
      python3 $(pwd)/database/$database_name/src/model-7/second_stage.py
      sub_menu;
      ;;

      23) clear;
      pyclean;
      option_picked "Model 8 - First Stage";
      python3 $(pwd)/database/$database_name/src/model-8/first_stage.py
      option_picked "Model 8 - Second Stage";
      python3 $(pwd)/database/$database_name/src/model-8/second_stage.py
      sub_menu;
      ;;

      30) clear;
      pyclean;
      option_picked "Model 9 - First Stage";
      python3 $(pwd)/database/$database_name/src/model-9/first_stage.py
      option_picked "Model 9 - Second Stage";
      python3 $(pwd)/database/$database_name/src/model-9/second_stage.py
      sub_menu;
      ;;

      31) clear;
      pyclean;
      option_picked "Model 10 - First Stage";
      python3 $(pwd)/database/$database_name/src/model-10/first_stage.py
      option_picked "Model 10 - Second Stage";
      python3 $(pwd)/database/$database_name/src/model-10/second_stage.py
      sub_menu;
      ;;

      32) clear;
      pyclean;
      option_picked "Model 11 - First Stage";
      python3 $(pwd)/database/$database_name/src/model-11/first_stage.py
      option_picked "Model 11 - Second Stage";
      python3 $(pwd)/database/$database_name/src/model-11/second_stage.py
      sub_menu;
      ;;

      33) clear;
      pyclean;
      option_picked "Model 12 - First Stage";
      python3 $(pwd)/database/$database_name/src/model-12/first_stage.py
      option_picked "Model 12 - Second Stage";
      python3 $(pwd)/database/$database_name/src/model-12/second_stage.py
      sub_menu;
      ;;

      x)exit;
      ;;

      \n)exit;
      ;;

      *)clear;
      option_picked "Pick an option from the menu";
      sub_menu;
      ;;
      esac
    fi
  done
}

clear
show_menu
