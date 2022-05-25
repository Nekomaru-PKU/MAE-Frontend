export function ImageUpload(props: {
    onImageUpload: (image: File) => void;
}) {
    return <div>
        <input type="file" accept=".png, .jpg" onChange={
            event => {
                if (event.target.files &&
                    event.target.files.length > 0) {
                    props.onImageUpload(event.target.files[0]!);
                }
        }} />
    </div>
}
