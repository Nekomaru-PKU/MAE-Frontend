import { Buffer } from 'buffer';
import { useContext, useState } from 'react';
import { css } from '@emotion/react';

import { MouseState } from './MouseStateProvider';

export function MaskEditor(props: {
    uploadId: string;
    maskBase64: string;
    onChange: (maskBase64: string) => void;
}) {
    const mask = MaskBase64Decode(props.maskBase64);
    return <table cellSpacing={0} css={css({
        width: "600px",
        height: "600px",
        background: `url(/upload/${props.uploadId})`,
        backgroundSize: 'cover',
    })}><tbody>{
        Array(14).fill(0).map((_, i) => <tr key={`tr-${i}`}>{
            Array(14).fill(0).map((_, j) =>
                <MaskCell
                    key={`tr-${i}-td-${j}`}
                    i={i} j={j}
                    state={mask[i]![j]!}
                    setState={(state: boolean) => {
                        mask[i]![j] = state;
                        props.onChange(MaskBase64Encode(mask));
                }} />)
        }</tr>)
    }</tbody>
    </table>
}

function MaskCell(props: {
    i: number,
    j: number,
    state: boolean,
    setState: (state: boolean) => void,
}) {
    const [hover, setHover] = useState(false);
    const mouseState = useContext(MouseState);
    return <td
        onContextMenu={event => event.preventDefault()}
        onMouseDown={event => {
            if (event.button === 0) props.setState(true);
            if (event.button === 2) props.setState(false);
            event.preventDefault();
        }}
        onMouseEnter={event => {
            setHover(true);
            if (mouseState.major) props.setState(true);
            if (mouseState.minor) props.setState(false);
            event.preventDefault();
        }}
        onMouseLeave={_ => setHover(false)}
        css={css({
            background: props.state ?
                hover ? 'rgba(0, 0, 0, 0.9)': 'rgba(0, 0, 0, 0.8)' :
                hover ? 'rgba(0, 0, 0, 0.2)': 'transparent',
        })}/>
}

function MaskBase64Encode(mask: boolean[][]) {
    const buf = Buffer.alloc(25, 0);
    for (let i = 0; i < 14; i++)
        for (let j = 0; j < 14; j++)
            if (mask[i]![j])
                buf[Math.floor((i * 14 + j) / 8)] |= 1 << (7 - (i * 14 + j) % 8);
    return buf.toString('base64');
}

function MaskBase64Decode(base64: string) {
    const buf = Buffer.from(base64, 'base64');
    return Array(14).fill(0).map((_, i) => Array(14).fill(0).map((_, j) =>
        (buf[Math.floor((i * 14 + j) / 8)]! & (1 << (7 - (i * 14 + j) % 8))) !== 0
    ));
}
