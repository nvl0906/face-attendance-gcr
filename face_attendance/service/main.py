from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.security import OAuth2PasswordBearer
from supabase import create_client, Client
from jose import jwt, JWTError
import io
from PIL import Image
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import os
import asyncio
from babel.dates import format_datetime
from dateutil.parser import parse
from typing import Optional
from urllib.parse import urlparse
import time
import unicodedata
from exponent_server_sdk import (
    DeviceNotRegisteredError,
    PushClient,
    PushMessage,
    PushServerError,
    PushTicketError,
)
from typing import List, Dict, Any
import uvicorn

# Load .env
load_dotenv()

GPS_ID = "00000000-0000-0000-0000-000000000001"
DIST_ID = "00000000-0000-0000-0000-000000000002"

# Initialize Supabase
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
JWT_SECRET = os.environ.get("JWT_SECRET")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM")
supabase: Client = create_client(url, key)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Initialize FastAPI
app = FastAPI(default_response_class=ORJSONResponse)

# CORS (optional for mobile app testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Notification Service
class NotificationService:
    """
    Service class for handling push notifications
    """
    
    @staticmethod
    def validate_expo_token(token: str) -> bool:
        """
        Validates that a token is a proper Expo push token
        """
        return PushClient().is_exponent_push_token(token)
    
    @staticmethod
    def register_device(member_id: str, expo_push_token: str, device_type: str, device_name: str = None) -> dict:
        """
        Registers a device for push notifications
        
        Args:
            member_id: User's member ID
            expo_push_token: Expo push token from device
            device_type: 'ios' or 'android'
            device_name: Optional device name
            
        Returns:
            dict: Registration result
        """
        # Validate token format
        if not NotificationService.validate_expo_token(expo_push_token):
            raise ValueError("Invalid Expo push token format")
        
        try:
            # Check if device already exists
            existing = supabase.table("user_devices").select("*").eq(
                "expo_push_token", expo_push_token
            ).execute()
            
            if existing.data:
                # Update existing device
                result = supabase.table("user_devices").update({
                    "member_id": member_id,
                    "device_type": device_type,
                    "device_name": device_name,
                    "is_active": True,
                }).eq("expo_push_token", expo_push_token).execute()
                
                return {
                    "success": True,
                    "message": "Device updated successfully",
                    "device_id": existing.data[0]["id"]
                }
            else:
                # Create new device record
                result = supabase.table("user_devices").insert({
                    "member_id": member_id,
                    "expo_push_token": expo_push_token,
                    "device_type": device_type,
                    "device_name": device_name,
                    "is_active": True
                }).execute()
                
                return {
                    "message": "Appareil enregistré avec succès",
                }
                
        except Exception as e:
            raise Exception(f"Failed to register device: {str(e)}")
    
    @staticmethod
    def unregister_device(member_id: str, expo_push_token: str) -> dict:
        """
        Unregisters a device (sets is_active to False)
        
        Args:
            member_id: User's member ID
            expo_push_token: Expo push token to unregister
            
        Returns:
            dict: Unregistration result
        """
        try:
            result = supabase.table("user_devices").update({
                "is_active": False,
            }).eq("expo_push_token", expo_push_token).eq(
                "member_id", member_id
            ).execute()
            
            return {
                "success": True,
                "message": "Device unregistered successfully"
            }
        except Exception as e:
            raise Exception(f"Failed to unregister device: {str(e)}")
    
    @staticmethod
    def get_user_devices(member_id: str) -> List[str]:
        """
        Retrieves all active Expo push tokens for a user
        
        Args:
            member_id: User's member ID
            
        Returns:
            List[str]: List of active Expo push tokens
        """
        try:
            result = supabase.table("user_devices").select("expo_push_token").eq(
                "member_id", member_id
            ).eq("is_active", True).execute()
            
            return [device["expo_push_token"] for device in result.data]
        except Exception as e:
            print(f"Error fetching user devices: {e}")
            return []
    
    @staticmethod
    def send_push_notification(
        tokens: List[str], 
        title: str, 
        body: str, 
        data: Dict[str, Any] = None
    ) -> Dict[str, int]:
        """
        Sends push notifications to multiple devices
        
        Args:
            tokens: List of Expo push tokens
            title: Notification title
            body: Notification body
            data: Optional data payload
            
        Returns:
            dict: {'sent': int, 'failed': int}
        """
        if not tokens:
            return {"sent": 0, "failed": 0}
        
        messages = []
        invalid_tokens = []
        
        # Create PushMessage objects for each token
        for token in tokens:
            if not PushClient().is_exponent_push_token(token):
                invalid_tokens.append(token)
                continue
            
            messages.append(PushMessage(
                to=token,
                title=title,
                body=body,
                data=data or {},
                sound='default',
                priority='high',
                channel_id='default',
            ))
        
        # Remove invalid tokens from database
        if invalid_tokens:
            NotificationService._deactivate_tokens(invalid_tokens)
        
        sent_count = 0
        failed_count = 0
        
        # Send notifications in chunks of 100
        chunk_size = 100
        for i in range(0, len(messages), chunk_size):
            chunk = messages[i:i + chunk_size]
            
            try:
                tickets = PushClient().publish_multiple(chunk)
                
                # Check for errors in tickets
                for ticket in tickets:
                    if ticket.is_success():
                        sent_count += 1
                    else:
                        failed_count += 1
                        if isinstance(ticket.error, DeviceNotRegisteredError):
                            print(f"Device not registered: {ticket.push_message.to}")
                            
            except PushServerError as exc:
                print(f"Expo push server error: {exc}")
                failed_count += len(chunk)
            except Exception as exc:
                print(f"Error sending notifications: {exc}")
                failed_count += len(chunk)
        
        return {"sent": sent_count, "failed": failed_count}
    
    @staticmethod
    def send_to_user(
        member_id: str,
        title: str,
        body: str,
        data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Sends notification to all of a user's active devices
        
        Args:
            member_id: User's member ID
            title: Notification title
            body: Notification body
            data: Optional data payload
            
        Returns:
            dict: Result with sent/failed counts
        """
        tokens = NotificationService.get_user_devices(member_id)
        
        if not tokens:
            return {
                "success": False,
                "message": "No active devices found for user",
                "sent": 0,
                "failed": 0
            }
        
        result = NotificationService.send_push_notification(tokens, title, body, data)
        
        return {
            "success": True,
            "message": f"Notifications sent to {len(tokens)} device(s)",
            "sent": result["sent"],
            "failed": result["failed"]
        }
    
    @staticmethod
    def _deactivate_tokens(tokens: List[str]):
        """
        Marks device tokens as inactive in the database
        """
        try:
            for token in tokens:
                supabase.table("user_devices").update({
                    "is_active": False,
                }).eq("expo_push_token", token).execute()
        except Exception as e:
            print(f"Error deactivating tokens: {e}")


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {"userid": payload.get("userid"), "username": payload.get("username"), "is_admin": payload.get("is_admin"), "voice": payload.get("voice"), "profile": payload.get("profile")}
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

async def verify_token(token: str):
    """Decode JWT token and raise if invalid"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return None

def normalizeword(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(
        char for char in normalized
        if unicodedata.category(char) != "Mn"
    )


async def get_emplacement():
    tz = timezone(timedelta(hours=3))
    now = datetime.now(tz)
    start_of_day = datetime.combine(now.date(), datetime.min.time())
    end_of_day = start_of_day + timedelta(days=1)

    response = await asyncio.to_thread(lambda: supabase.table("attendance")
        .select("emplacement, timestamp")
        .gte("timestamp", start_of_day.isoformat())
        .lt("timestamp", end_of_day.isoformat())
        .limit(1)
        .execute()
    )

    data = response.data
    if data and len(data) > 0:
        value = data[0]["emplacement"]
    else:
        value = "aucun"
    return value

async def get_all_members():
    """Use the view instead of computing in Python"""
    response = await asyncio.to_thread(
        lambda: supabase.table("member_statistics")  # Use the view!
        .select("*")
        .execute()
    )
    return response.data if response else []

async def get_user_attendance(member_id: str):
    """Use the Postgres function"""
    response = await asyncio.to_thread(
        lambda: supabase.rpc('get_user_attendance', {'user_id': member_id})
        .execute()
    )
    if response and response.data:
        # Add formatted timestamp to each record
        for record in response.data:
            ts_parsed = parse(record['timestamp_raw'])
            record['timestamp'] = format_datetime(
                ts_parsed, 
                "EEEE d MMMM y", 
                locale="mg_MG"
            )
    return response.data if response else []

async def get_update_attendance(userId, emplacement, timestamp):

    user_attendance = await get_user_attendance(userId)
    
    # Find and update the matching record
    for record in user_attendance:
        if record["emplacement"] == emplacement and record["timestamp"] == timestamp:
            record["attendance"] = "present"
            await asyncio.to_thread(lambda: supabase.table("attendance").insert({
                "member_id": userId,
                "emplacement":  emplacement,
                "timestamp": record["timestamp_raw"]
            }).execute())

    return True

async def get_users():
    response = await asyncio.to_thread(lambda: supabase.table("members").select("username").execute().data)
    return [user["username"] for user in response] if response else []

async def verify_admin(userId: str, admin: bool):
    user = await asyncio.to_thread(lambda: supabase.table("members").select("is_admin").eq("id", userId).execute().data[0])
    if admin == user["is_admin"]:
        return True
    else:
        return False

@app.post("/v2/delete-user")
async def delete_user(payload: dict = Body(...), current_user=Depends(get_current_user)):
    admin_check = await verify_admin(current_user.get("userid"), current_user.get("is_admin"))
    if not admin_check:
        return {"status":"errorAdmin", "message":"Veuillez vous-reconnecter svp!"}

    user_id = payload.get("id")
    user_name = payload.get("name")

    # Remove existing photo from Supabase if exists
    existing_url = await asyncio.to_thread(
        lambda: supabase.table("members").select("profile").eq("username", user_name).execute().data[0].get("profile")
    )
    if existing_url:
        parsed_url = urlparse(existing_url)
        
        # Extract path AFTER the bucket name (images/)
        full_path = parsed_url.path
        bucket_prefix = "/storage/v1/object/public/images/"
        if bucket_prefix in full_path:
            existing_path = full_path.split(bucket_prefix, 1)[1]
            await asyncio.to_thread(
                lambda: supabase.storage
                    .from_("images")
                    .remove([existing_path])
            )

    # Delete from Supabase
    await asyncio.to_thread(lambda: supabase.table("members").delete().eq("id", user_id).execute())

    return {"status": "success", "message": f"{user_name} supprimé avec succès!"}

@app.post("/v2/update-user")
async def update_user(payload: dict = Body(...), current_user=Depends(get_current_user)):
    admin_check = await verify_admin(current_user.get("userid"), current_user.get("is_admin"))
    if not admin_check:
        return {"status":"errorAdmin", "message":"Veuillez vous-reconnecter svp!"}

    user_id = payload.get("id")
    new_name = payload.get("name")
    new_voice = payload.get("voice")
    new_admin = payload.get("is_admin")
    res = await asyncio.to_thread(lambda: supabase.table("members").select("*").eq("id", user_id).execute().data[0])

    if new_name == res["username"] and new_voice == res["voice"] and new_admin == res["is_admin"] :
        return {"status": "error", "message": "Aucun changement détecté!"}
    
    # Update in Supabase
    if new_name != res["username"]:
        users = await get_users()
        if new_name in users:
            return {"status": "error", "message": f"{new_name} est déjà pris!"}
        
        existing_url = res["profile"]
        if existing_url:
            parsed_url = urlparse(existing_url)
            
            # Extract path AFTER the bucket name (images/)
            full_path = parsed_url.path
            bucket_prefix = "/storage/v1/object/public/images/"
            if bucket_prefix in full_path:
                existing_path = full_path.split(bucket_prefix, 1)[1]
                supabase_path = f"profile/{normalizeword(new_name)}_{int(time.time())}.jpg"
                await asyncio.to_thread(
                    lambda: supabase.storage
                        .from_("images")
                        .move(existing_path, supabase_path)
                )

            public_url = await asyncio.to_thread(
                lambda: supabase.storage.from_('images').get_public_url(supabase_path)
            )
            if public_url:
                resp = await asyncio.to_thread(lambda: supabase.table("members").update({"username": new_name}).eq("id", user_id).execute())
                if resp:
                    await asyncio.to_thread(
                        lambda: supabase.table("members").update({"profile": public_url}).eq("username", new_name).execute()
                    )

    if new_voice != res["voice"]:
        await asyncio.to_thread(lambda: supabase.table("members").update({"voice": new_voice}).eq("id", user_id).execute())

    if new_admin != res["is_admin"]:
        await asyncio.to_thread(lambda: supabase.table("members").update({"is_admin": new_admin}).eq("id", user_id).execute())


    return {"status": "success", "message": f"{new_name} mis à jour avec succès!"}

@app.get("/v2/emplacement")
async def get_current_emplacement(current_user=Depends(get_current_user)):
    admin_check = await verify_admin(current_user.get("userid"), current_user.get("is_admin"))
    if not admin_check:
        return {"status":"errorAdmin", "message":"Veuillez vous-reconnecter svp!"}
    value = await get_emplacement()
    return {"sup_emplacement": value}

@app.get("/v2/mypresence")
async def user_attendance(current_user=Depends(get_current_user)):
    admin_check = await verify_admin(current_user.get("userid"), current_user.get("is_admin"))
    if not admin_check:
        return {"status":"errorAdmin", "message":"Veuillez vous-reconnecter svp!"}
    value = await get_user_attendance(current_user.get("userid"))
    return {"status":"success", "message":"success", "mypresence": value}

@app.get("/v2/allusers")
async def all_users(current_user=Depends(get_current_user)):
    admin_check = await verify_admin(current_user.get("userid"), current_user.get("is_admin"))
    if not admin_check:
        return {"status":"errorAdmin", "message":"Veuillez vous-reconnecter svp!"}

    value = await get_all_members()
    if value:
        value.sort(key=lambda x: x['username'].lower())
        return {"allusers": value}
    else:
        return {"allusers": []}

@app.post("/v2/userpresence")
async def user_only_attendance(userId = Form(...), current_user=Depends(get_current_user)):
    admin_check = await verify_admin(current_user.get("userid"), current_user.get("is_admin"))
    if not admin_check:
        return {"status":"errorAdmin", "message":"Veuillez vous-reconnecter svp!"}

    value = await get_user_attendance(userId)
    return {"userpresence": value}

@app.post("/v2/updatepresence")
async def update_attendance(userId = Form(...), emplacement = Form(...), timestamp = Form(...), current_user=Depends(get_current_user)):
    admin_check = await verify_admin(current_user.get("userid"), current_user.get("is_admin"))
    if not admin_check:
        return {"status":"errorAdmin", "message":"Veuillez vous-reconnecter svp!"}

    value = await get_update_attendance(userId, emplacement, timestamp)
    if value:
        return {"message": "Succès"}

@app.post("/v2/gps")
async def set_gps_admin(latitude: float = Form(...), longitude: float = Form(...), current_user=Depends(get_current_user)):
    admin_check = await verify_admin(current_user.get("userid"), current_user.get("is_admin"))
    if not admin_check:
        return {"status":"errorAdmin", "message":"Veuillez vous-reconnecter svp!"}

    # Update GPS location for admin
    def upsert_gps():
        supabase.table("gps").upsert({"id": GPS_ID, "latitude": latitude, "longitude": longitude}).execute()

    await asyncio.to_thread(upsert_gps)
    return {"message":"GPS enregistré avec succès"}

@app.post("/v2/update-dist")
async def update_distance(payload: dict = Body(...), current_user=Depends(get_current_user)):
    admin_check = await verify_admin(current_user.get("userid"), current_user.get("is_admin"))
    if not admin_check:
        return {"status":"errorAdmin", "message":"Veuillez vous-reconnecter svp!"}

    dist = payload.get("dist")
    # Update GPS location for admin
    def upsert_distance():
        supabase.table("distance").upsert({"id": DIST_ID, "dist": dist}).execute()

    await asyncio.to_thread(upsert_distance)
    return {"message":f"Distance de {dist}m enregistré  avec succès"}

@app.get("/v2/get-dist")
async def update_distance(current_user=Depends(get_current_user)):
    admin_check = await verify_admin(current_user.get("userid"), current_user.get("is_admin"))
    if not admin_check:
        return {"status":"errorAdmin", "message":"Veuillez vous-reconnecter svp!"}

    # Update GPS location for admin
    def get_distance():
        dist = supabase.table("distance").select({"dist"}).execute().data[0]["dist"]
        return dist

    distance = await asyncio.to_thread(get_distance)
    return {"distance": distance}

@app.post("/v2/register-device")
async def register_device(
    expo_push_token: str = Form(...),
    device_type: str = Form(...),
    device_name: Optional[str] = Form(None),
    current_user=Depends(get_current_user)
):
    admin_check = await verify_admin(current_user.get("userid"), current_user.get("is_admin"))
    if not admin_check:
        return {"status":"errorAdmin", "message":"Veuillez vous-reconnecter svp!"}

    """
    Register a device for push notifications
    """
    try:
        member_id = current_user.get("userid")
        result = NotificationService.register_device(
            member_id=member_id,
            expo_push_token=expo_push_token,
            device_type=device_type,
            device_name=device_name
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v2/unregister-device")
async def unregister_device(
    expo_push_token: str = Form(...),
    current_user=Depends(get_current_user)
):
    admin_check = await verify_admin(current_user.get("userid"), current_user.get("is_admin"))
    if not admin_check:
        return {"status":"errorAdmin", "message":"Veuillez vous-reconnecter svp!"}
    """
    Unregister a device (on logout)
    """
    try:
        member_id = current_user.get("userid")
        result = NotificationService.unregister_device(member_id, expo_push_token)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v2/userphoto")
async def update_profile(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    admin_check = await verify_admin(current_user.get("userid"), current_user.get("is_admin"))
    if not admin_check:
        return {"status":"errorAdmin", "message":"Veuillez vous-reconnecter svp!"}

    """Upload/update user photo in known_faces folder"""
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    username = current_user["username"]
    supabase_path = f"profile/{normalizeword(username)}_{int(time.time())}.jpg"

    # Read file content ONCE
    raw_bytes = await file.read()

    # Resize to 100x100 using PIL (force exact size)
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        pil_img = pil_img.resize((200, 200), Image.LANCZOS)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=100)
        file_content = buf.getvalue()
    except Exception as e:
        # If resizing fails, fall back to original bytes but log
        print("⚠️ Resize failed, using original bytes:", e)
        file_content = raw_bytes

    # Remove existing photo from Supabase if exists
    existing_url = await asyncio.to_thread(
        lambda: supabase.table("members").select("profile").eq("username", username).execute().data[0].get("profile")
    )
    if existing_url:
        parsed_url = urlparse(existing_url)
        
        # Extract path AFTER the bucket name (images/)
        full_path = parsed_url.path
        bucket_prefix = "/storage/v1/object/public/images/"
        if bucket_prefix in full_path:
            existing_path = full_path.split(bucket_prefix, 1)[1]
            await asyncio.to_thread(
                lambda: supabase.storage
                    .from_("images")
                    .remove([existing_path])
            )

    # Upload to Supabase using the file content (bytes)
    response = await asyncio.to_thread(
        lambda: supabase.storage.from_('images').upload(
            supabase_path, 
            file_content, 
            file_options={"upsert": "false"}
        )
    )
    if response:
        public_url = await asyncio.to_thread(
            lambda: supabase.storage.from_('images').get_public_url(supabase_path)
        )
        if public_url:
            await asyncio.to_thread(
                lambda: supabase.table("members").update({"profile": public_url}).eq("username", username).execute()
            )

    return {"status": "success", "photoUrl": public_url, "message": "Photo de profil mise à jour avec succès"}    

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))