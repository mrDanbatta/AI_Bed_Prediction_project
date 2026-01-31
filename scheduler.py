from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import logging
from datetime import datetime
import pandas as pd
import os
from database import get_hospital_names, get_wards, get_ward_data
from preprocessing import preprocess_ward_data
from prediction import process_single_ward
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()

def update_all_models_and_data():
    """
    Update all models and data for all hospital-ward combinations every 14 days.
    """
    logger.info(f"Starting scheduled update at {datetime.now()}")
    
    try:
        hospital_names = get_hospital_names()
        
        for hospital in hospital_names:
            wards = get_wards(hospital)
            
            for ward in wards:
                logger.info(f"Updating data for {hospital} - {ward}")
                update_ward_data_and_model(hospital, ward)
        
        logger.info("Update cycle completed successfully")
        
    except Exception as e:
        logger.error(f"Error during update cycle: {str(e)}")

def update_ward_data_and_model(hospital, ward):
    """
    Update data and retrain model for a specific hospital-ward combination.
    
    Parameters:
    -----------
    hospital : str
        Hospital identifier
    ward : str
        Ward name
    """
    try:
        # Fetch fresh data from database
        logger.info(f"Fetching fresh data for {hospital} - {ward}")
        df = get_ward_data(hospital, ward)
        
        # Preprocess the data
        ts_data = preprocess_ward_data(df)
        
        # Delete old model to force retraining
        model_path = f'models/{hospital}_{ward}_sarima_model.pkl'
        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info(f"Deleted old model at {model_path}")
        
        # Retrain model with fresh data
        logger.info(f"Retraining model for {hospital} - {ward}")
        forecast_df, results_dict, train, test = process_single_ward(
            ts_data, 
            hospital, 
            ward
        )
        
        # Save update timestamp
        log_update(hospital, ward)
        logger.info(f"Successfully updated {hospital} - {ward}")
        
    except Exception as e:
        logger.error(f"Error updating {hospital} - {ward}: {str(e)}")

def log_update(hospital, ward):
    """
    Log the update timestamp for tracking purposes.
    
    Parameters:
    -----------
    hospital : str
        Hospital identifier
    ward : str
        Ward name
    """
    log_file = 'data/update_log.csv'
    update_record = pd.DataFrame({
        'hospital': [hospital],
        'ward': [ward],
        'last_updated': [datetime.now()]
    })
    
    if os.path.exists(log_file):
        existing_log = pd.read_csv(log_file)
        # Remove existing record for this hospital-ward combination
        existing_log = existing_log[~((existing_log['hospital'] == hospital) & (existing_log['ward'] == ward))]
        update_record = pd.concat([existing_log, update_record], ignore_index=True)
    
    update_record.to_csv(log_file, index=False)

def start_scheduler():
    """
    Start the background scheduler for 14-day updates.
    """
    if not scheduler.running:
        # Schedule to run every 14 days
        scheduler.add_job(
            update_all_models_and_data,
            trigger=IntervalTrigger(days=14),
            id='update_models_14d',
            name='Update all models and data every 14 days',
            replace_existing=True
        )
        scheduler.start()
        logger.info("Scheduler started - updates scheduled every 14 days")

def stop_scheduler():
    """
    Stop the background scheduler.
    """
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler stopped")

if __name__ == '__main__':
    start_scheduler()