@echo
python process_data.py disaster_messages.csv disaster_categories.csv sqlite:///DisasterResponse.db
pause