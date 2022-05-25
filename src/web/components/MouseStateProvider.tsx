import { createContext, useState, useEffect } from "react";

export const MouseState = createContext({
    major: false,
    minor: false,
});

export function MouseStateProvider(props: { children: React.ReactNode }) {
    const [major, setMajor] = useState(false);
    const [minor, setMinor] = useState(false);
    useEffect(() => {
        const onMouseDown = (event: MouseEvent) => {
            if (event.button === 0) setMajor(true);
            if (event.button === 2) setMinor(true);
        };
        const onMouseUp = (event: MouseEvent) => {
            if (event.button === 0) setMajor(false);
            if (event.button === 2) setMinor(false);
        };
        document.addEventListener('mousedown', onMouseDown);
        document.addEventListener('mouseup'  , onMouseUp);
        return () => {
            document.removeEventListener('mousedown', onMouseDown);
            document.removeEventListener('mouseup'  , onMouseUp);
        }
    })
    return <MouseState.Provider value={{ major, minor }} children={props.children} />
}
