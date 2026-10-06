# HiFi-MobiNet visual identity

The original image files stay in ignored `visual-branding/`.
Do not stage this folder or force-add its images.
Only image URLs, source names, sizes and hashes are tracked in Git.

Hugging Face Space hosts three unchanged author-supplied PNG files.
The site header uses the horizontal logo. The hero and social metadata use the cover.
The GitHub README uses the repository banner.
GitHub repository social preview also uses this banner, stored as a platform setting outside Git.
The Space card uses the cover and a custom thumbnail URL.
Both listening sites reference the same pinned asset revision.

The identity board and two white symbol variants remain local.
They are not required for the published interface.
No image was edited or converted, and no new artwork license was assigned.
Frontend code remains under its existing MIT scope.

See [publication record](branding-publication.json) for pinned URLs and byte identities.
Audio catalog, model artifacts and research claims remain unchanged.

To update branding, upload only the approved assets to the existing Space.
Record the new asset commit, sizes and hashes.
Update the site and card URLs, then verify both deployments.
Keep image files outside Git history.

## Completed checks

Both deployments serve the updated frontend. Three anonymous branding downloads matched source hashes.
Logo and cover loaded in the live browser; the GitHub README banner also loaded.
The Space thumbnail metadata matches the pinned cover URL.
Desktop and mobile previews passed visual checks. The existing audio player still works.
The source artwork and audio catalog were not modified.
