import * as ReactDOM from "react-dom";
import { css } from '@emotion/react';

import { CssBaseline, ThemeProvider } from "@mui/material";
import { createTheme } from "@mui/material/styles"

import { AppContent } from "./content";
import { MouseStateProvider } from "./components/MouseStateProvider";

const theme = createTheme({
    palette: {
        background: {
            default: "#E3F2FD",
        }
    }
});

function App() {;
    return <ThemeProvider theme={theme}>
        <CssBaseline />
        <MouseStateProvider>
            <div css={css({
                display: 'flex',
                justifyContent: 'center',
                // fontSize: "calc(min(2vh, 2vw))",
            })}><AppContent /></div>
        </MouseStateProvider>
    </ThemeProvider>
}

ReactDOM.render(<App />, document.getElementById("root"));
