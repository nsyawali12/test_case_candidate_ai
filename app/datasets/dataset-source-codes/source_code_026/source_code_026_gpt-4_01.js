import React, { useState, useEffect, memo } from "react";

// Wrapping the Component with React.memo to prevent unnecessary rerenders
const MemoComponent = memo(function MemoComponent({ data }) {
    // Only rerenders if data prop changes

    return (
        <div>
            {data.map((item, index) => (
                <p key={index}>{item}</p>
            ))}
        </div>
    );
});

function MainComponent() {
    const [data, setData] = useState([]);
    const [otherState, setOtherState] = useState("");

    // Mimic data fetch
    useEffect(() => {
        fetchSomeData().then(fetchedData => {
            setData(fetchedData);
        });
    }, []);

    return (
        <div>
            {/* This component only rerender when 'data' changes */}
            <MemoComponent data={data} />

            {/* otherState changes do not cause MemoComponent rerenders */}
            <button onClick={() => setOtherState(Math.random().toString())}>
                Change Other State
            </button>
        </div>
    );
}

async function fetchSomeData() {
    // Simulate data fetching
    await new Promise(r => setTimeout(r, 2000));
    return ["Data1", "Data2", "Data3"];
}

export default MainComponent;
