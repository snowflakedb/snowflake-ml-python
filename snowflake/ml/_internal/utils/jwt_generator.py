import base64
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import types

logger = logging.getLogger(__name__)

ISSUER = "iss"
EXPIRE_TIME = "exp"
ISSUE_TIME = "iat"
SUBJECT = "sub"


class JWTGenerator:
    """
    Creates and signs a JWT with the specified private key file, username, and account identifier. The JWTGenerator
    keeps the generated token and only regenerates the token if a specified period of time has passed.
    """

    _DEFAULT_LIFETIME = timedelta(minutes=59)  # The tokens will have a 59-minute lifetime
    _DEFAULT_RENEWAL_DELTA = timedelta(minutes=54)  # Tokens will be renewed after 54 minutes
    ALGORITHM = "RS256"  # Tokens will be generated using RSA with SHA256

    def __init__(
        self,
        account: str,
        user: str,
        private_key: types.PRIVATE_KEY_TYPES,
        lifetime: Optional[timedelta] = None,
        renewal_delay: Optional[timedelta] = None,
    ) -> None:
        """
        Create a new JWTGenerator object.

        Args:
            account: The account identifier.
            user: The username.
            private_key: The private key used to sign the JWT.
            lifetime: The lifetime of the token.
            renewal_delay: The time before the token expires to renew it.
        """

        # Construct the fully qualified name of the user in uppercase.
        self.account = JWTGenerator._prepare_account_name_for_jwt(account)
        self.user = user.upper()
        self.qualified_username = self.account + "." + self.user
        self.private_key = private_key
        self.public_key_fp = JWTGenerator._calculate_public_key_fingerprint(self.private_key)

        self.issuer = self.qualified_username + "." + self.public_key_fp
        self.lifetime = lifetime or JWTGenerator._DEFAULT_LIFETIME
        self.renewal_delay = renewal_delay or JWTGenerator._DEFAULT_RENEWAL_DELTA
        self.renew_time = datetime.now(timezone.utc)
        self.token: Optional[str] = None

        logger.info(
            """Creating JWTGenerator with arguments
            account : %s, user : %s, lifetime : %s, renewal_delay : %s""",
            self.account,
            self.user,
            self.lifetime,
            self.renewal_delay,
        )

    @staticmethod
    def _prepare_account_name_for_jwt(raw_account: str) -> str:
        account = raw_account
        if ".global" not in account:
            # Handle the general case.
            idx = account.find(".")
            if idx > 0:
                account = account[0:idx]
        else:
            # Handle the replication case.
            idx = account.find("-")
            if idx > 0:
                account = account[0:idx]
        # Use uppercase for the account identifier.
        return account.upper()

    def get_token(self) -> str:
        now = datetime.now(timezone.utc)  # Fetch the current time
        if self.token is not None and self.renew_time > now:
            return self.token

        # If the token has expired or doesn't exist, regenerate the token.
        logger.info(
            "Generating a new token because the present time (%s) is later than the renewal time (%s)",
            now,
            self.renew_time,
        )
        # Calculate the next time we need to renew the token.
        self.renew_time = now + self.renewal_delay

        # Create our payload
        payload = {
            # Set the issuer to the fully qualified username concatenated with the public key fingerprint.
            ISSUER: self.issuer,
            # Set the subject to the fully qualified username.
            SUBJECT: self.qualified_username,
            # Set the issue time to now.
            ISSUE_TIME: now,
            # Set the expiration time, based on the lifetime specified for this object.
            EXPIRE_TIME: now + self.lifetime,
        }

        # Regenerate the actual token
        token = jwt.encode(payload, key=self.private_key, algorithm=JWTGenerator.ALGORITHM)  # type: ignore[arg-type]
        # If you are using a version of PyJWT prior to 2.0, jwt.encode returns a byte string instead of a string.
        # If the token is a byte string, convert it to a string.
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        self.token = token
        public_key = self.private_key.public_key()
        logger.info(
            "Generated a JWT with the following payload: %s",
            jwt.decode(self.token, key=public_key, algorithms=[JWTGenerator.ALGORITHM]),  # type: ignore[arg-type]
        )

        return token

    @staticmethod
    def _calculate_public_key_fingerprint(private_key: types.PRIVATE_KEY_TYPES) -> str:
        # Get the raw bytes of public key.
        public_key_raw = private_key.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # Get the sha256 hash of the raw bytes.
        sha256hash = hashlib.sha256()
        sha256hash.update(public_key_raw)

        # Base64-encode the value and prepend the prefix 'SHA256:'.
        public_key_fp = "SHA256:" + base64.b64encode(sha256hash.digest()).decode("utf-8")
        logger.info("Public key fingerprint is %s", public_key_fp)

        return public_key_fp
