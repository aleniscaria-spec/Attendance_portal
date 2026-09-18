
import os

import cv2
import numpy as np
import face_recognition


class FaceRecognitionEngine:
    """
    Face Recognition Engine

    Responsibilities:
    1. Load registered student faces
    2. Generate face embeddings
    3. Detect faces in incoming images
    4. Compare embeddings
    5. Return recognized identities
    """

    def __init__(
        self,
        dataset_dir="dataset",
        tolerance=0.50
    ):

        self.dataset_dir = dataset_dir
        self.tolerance = tolerance

        self.known_encodings = []
        self.known_names = []

        self.load_database()


    # ========================================================
    # LOAD FACE DATABASE
    # ========================================================

    def load_database(self):
        """
        Load all registered student images
        and generate their face embeddings.
        """

        self.known_encodings = []
        self.known_names = []

        if not os.path.exists(
            self.dataset_dir
        ):

            os.makedirs(
                self.dataset_dir,
                exist_ok=True
            )

            return


        for filename in os.listdir(
            self.dataset_dir
        ):

            if not filename.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            ):

                continue


            image_path = os.path.join(
                self.dataset_dir,
                filename
            )


            try:

                # --------------------------------------------
                # Load image
                # --------------------------------------------

                image = (
                    face_recognition
                    .load_image_file(
                        image_path
                    )
                )


                # --------------------------------------------
                # Detect faces
                # --------------------------------------------

                face_locations = (
                    face_recognition
                    .face_locations(
                        image
                    )
                )


                if len(face_locations) == 0:

                    print(
                        f"[WARNING] "
                        f"No face found: {filename}"
                    )

                    continue


                # --------------------------------------------
                # Generate face embedding
                # --------------------------------------------

                encodings = (
                    face_recognition
                    .face_encodings(
                        image,
                        face_locations
                    )
                )


                if len(encodings) == 0:

                    print(
                        f"[WARNING] "
                        f"Could not encode: {filename}"
                    )

                    continue


                # Use first face
                encoding = encodings[0]


                # --------------------------------------------
                # Extract student name
                # --------------------------------------------

                name = os.path.splitext(
                    filename
                )[0]

                name = name.replace(
                    "_",
                    " "
                )


                # --------------------------------------------
                # Store embedding
                # --------------------------------------------

                self.known_encodings.append(
                    encoding
                )

                self.known_names.append(
                    name
                )


                print(
                    f"[ML] Loaded: {name}"
                )


            except Exception as error:

                print(
                    f"[ML ERROR] "
                    f"{filename}: {error}"
                )


        print(
            f"[ML] Database contains "
            f"{len(self.known_names)} student(s)"
        )


    # ========================================================
    # RECOGNIZE FACE
    # ========================================================

    def recognize(
        self,
        image
    ):
        """
        Recognize all faces in an RGB image.

        Parameters
        ----------
        image : numpy.ndarray
            RGB image.

        Returns
        -------
        list
            Recognition results.
        """


        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        face_locations = (
            face_recognition
            .face_locations(
                image,
                model="hog"
            )
        )


        # ----------------------------------------------------
        # Generate embeddings
        # ----------------------------------------------------

        face_encodings = (
            face_recognition
            .face_encodings(
                image,
                face_locations
            )
        )


        results = []


        # ----------------------------------------------------
        # Process every detected face
        # ----------------------------------------------------

        for encoding, location in zip(
            face_encodings,
            face_locations
        ):

            name = "Unknown Student"

            recognized = False

            confidence = 0.0

            distance = None


            # ------------------------------------------------
            # Compare with database
            # ------------------------------------------------

            if len(
                self.known_encodings
            ) > 0:


                distances = (
                    face_recognition
                    .face_distance(
                        self.known_encodings,
                        encoding
                    )
                )


                # Smallest distance
                best_index = int(
                    np.argmin(
                        distances
                    )
                )


                distance = float(
                    distances[
                        best_index
                    ]
                )


                # --------------------------------------------
                # Convert distance into approximate score
                # --------------------------------------------

                confidence = max(
                    0.0,
                    min(
                        1.0,
                        1.0 - distance
                    )
                )


                # --------------------------------------------
                # Recognition decision
                # --------------------------------------------

                if distance <= self.tolerance:

                    name = (
                        self.known_names[
                            best_index
                        ]
                    )

                    recognized = True


            results.append({

                "name": name,

                "recognized":
                    recognized,

                "confidence":
                    round(
                        confidence * 100,
                        2
                    ),

                "distance":
                    round(
                        distance,
                        4
                    )
                    if distance is not None
                    else None,

                "location":
                    list(location)

            })


        return results


    # ========================================================
    # REGISTER FACE
    # ========================================================

    def validate_registration_image(
        self,
        image
    ):
        """
        Validate that a registration photo
        contains exactly one face.

        Returns
        -------
        tuple
            (success, message)
        """


        locations = (
            face_recognition
            .face_locations(
                image
            )
        )


        if len(locations) == 0:

            return (
                False,
                "No face detected."
            )


        if len(locations) > 1:

            return (
                False,
                "Multiple faces detected."
            )


        encodings = (
            face_recognition
            .face_encodings(
                image,
                locations
            )
        )


        if len(encodings) == 0:

            return (
                False,
                "Could not generate face encoding."
            )


        return (
            True,
            "Face is valid."
        )


    # ========================================================
    # RELOAD
    # ========================================================

    def reload(self):

        print(
            "[ML] Reloading face database..."
        )

        self.load_database()


    # ========================================================
    # DATABASE INFORMATION
    # ========================================================

    def student_count(self):

        return len(
            self.known_names
        )


    def student_names(self):

        return list(
            self.known_names
        )

