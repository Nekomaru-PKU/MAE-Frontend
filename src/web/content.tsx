import { useState } from 'react';
import { css } from '@emotion/react'
import { Button, Collapse, Step, StepContent, StepLabel, Stepper, useTheme } from "@mui/material";
import { ImageUpload } from './components/ImageUpload';
import { MaskEditor } from './components/MaskEditor';
import { ResultViewer } from './components/ResultViewer';
import { ModelSwitch } from './components/ModelSwitch';
import styled from '@emotion/styled';

const MyButton = styled(Button)({
    height: 28,
    marginLeft: 8,
    borderRadius: 14,
});

function MyButtonContainer(props: {
    children: React.ReactNode;
}) {
    return <div css={css({
        float: 'right',
        marginTop: 8,
    })}>{props.children}</div>;
}

export function AppContent() {
    const theme = useTheme();
    const [step, setStep] = useState(0);
    const [modelId, setModelId] = useState(0);
    const [uploadId, setUploadId] = useState('');
    const [maskBase64, setMaskBase64] = useState("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==");
    return <div css={css({
        width: "640px",
        marginTop: 64,
        marginBottom: 64,
    })}>
        <div>
            <h1 css={css({
                color: theme.palette.primary.main,
                display: 'inline',
            })}>MAE Inpainting</h1>
            <ModelSwitch modelId={modelId} onChange={setModelId} />
        </div>
        <Stepper activeStep={step} orientation="vertical">
            <Step>
                <StepLabel>Upload Image</StepLabel>
                <StepContent>
                    <ImageUpload onImageUpload={async image => {
                        const formData = new FormData();
                        formData.append('image', image, image.name);
                        const res = await fetch('/api/upload', {
                            method: 'POST',
                            body: formData,
                        });
                        const json = await res.json();
                        setUploadId(json.id);
                        setStep(1);
                        console.log(`Upload ID: ${json.id}`);
                    }}/>
                </StepContent>
            </Step>
            <Step>
                <StepLabel>Paint Mask</StepLabel>
                <StepContent>{ uploadId ? <>
                    <MaskEditor
                        uploadId={uploadId}
                        maskBase64={maskBase64}
                        onChange={newMaskBase64 => setMaskBase64(newMaskBase64)} />
                    <MyButtonContainer>
                        <MyButton variant='outlined'  onClick={() => setStep(0)}>Back</MyButton>
                        <MyButton variant='contained' onClick={() => {
                            fetch("/api/run", {
                                method: "POST",
                                headers: {
                                'Content-Type': 'application/json'
                                },
                                body: JSON.stringify({
                                    id: uploadId,
                                    modelId: modelId.toString(),
                                    maskBase64: maskBase64,
                                })
                            });
                            setStep(2);
                        }}>Next</MyButton>
                    </MyButtonContainer>
                </> : (<></>) }</StepContent>
            </Step>
            <Step>
                <StepLabel>Reconstruct</StepLabel>
                <StepContent>
                    <ResultViewer uploadId={uploadId} />
                    <MyButtonContainer>
                        <MyButton variant='outlined'  onClick={() => setStep(0)}>Back</MyButton>
                    </MyButtonContainer>
                </StepContent>
            </Step>
        </Stepper>
    </div>
}
