`timescale 1ns/1ps

module nn_accelerator_tb;

    parameter int NUM_SAMPLES = 360;
    parameter int NUM_TESTS = 100;
    parameter int INPUT_SIZE = 64;
    parameter int HIDDEN_SIZE = 16;
    parameter int OUTPUT_SIZE = 10;
    parameter int TIMEOUT_CYCLES = 5000;

    logic signed [7:0] all_inputs [0:NUM_SAMPLES*INPUT_SIZE-1];
    logic signed [31:0] expected_predictions [0:NUM_SAMPLES-1];
    logic signed [31:0] expected_hidden_all [0:NUM_SAMPLES*HIDDEN_SIZE-1];
    logic signed [31:0] expected_output_all [0:NUM_SAMPLES*OUTPUT_SIZE-1];

    logic clk;
    logic rst;
    logic start;
    logic done;
    logic [3:0] prediction;

    int prediction_errors;
    int hidden_errors;
    int output_errors;
    int sample_errors;
    int total_errors;
    int cycle_count;
    int total_cycles;

    nn_accelerator dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .done(done),
        .prediction(prediction)
    );

    always #5 clk = ~clk;

    initial begin
        if (NUM_TESTS > NUM_SAMPLES) begin
            $fatal(1, "NUM_TESTS (%0d) exceeds NUM_SAMPLES (%0d)",
                NUM_TESTS, NUM_SAMPLES);
        end

        $readmemh("data/mem/inputs.mem", all_inputs);
        $readmemh("data/mem/expected_predictions.mem", expected_predictions);
        $readmemh("data/mem/expected_hidden_all.mem", expected_hidden_all);
        $readmemh("data/mem/expected_output_all.mem", expected_output_all);

        clk = 1'b0;
        rst = 1'b1;
        start = 1'b0;

        prediction_errors = 0;
        hidden_errors = 0;
        output_errors = 0;
        total_errors = 0;
        total_cycles = 0;

        for (int sample = 0; sample < NUM_TESTS; sample++) begin
            sample_errors = 0;
            cycle_count = 0;

            // The DUT remains in DONE_STATE after each run, so reset it before
            // loading the next input sample.
            @(negedge clk);
            rst = 1'b1;
            start = 1'b0;
            repeat (2) @(posedge clk);

            // inputs.mem is sample-major. Copy the selected sample's 64
            // values into the DUT input memory before starting inference.
            @(negedge clk);
            for (int i = 0; i < INPUT_SIZE; i++) begin
                dut.input_mem[i] = all_inputs[sample*INPUT_SIZE + i];
            end
            rst = 1'b0;

            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            while ((done !== 1'b1) && (cycle_count < TIMEOUT_CYCLES)) begin
                @(posedge clk);
                cycle_count++;
            end

            if (done !== 1'b1) begin
                $display("Sample %0d: TIMEOUT after %0d cycles", sample, cycle_count);
                sample_errors++;
            end else begin
                if (prediction !== expected_predictions[sample][3:0]) begin
                    $display(
                        "Sample %0d prediction mismatch: RTL=%0d expected=%0d",
                        sample, prediction, expected_predictions[sample]
                    );
                    prediction_errors++;
                    sample_errors++;
                end

                for (int i = 0; i < HIDDEN_SIZE; i++) begin
                    if (
                        dut.hidden_mem[i] !==
                        expected_hidden_all[sample*HIDDEN_SIZE + i]
                    ) begin
                        $display(
                            "Sample %0d hidden[%0d] mismatch: RTL=%0d expected=%0d",
                            sample, i, dut.hidden_mem[i],
                            expected_hidden_all[sample*HIDDEN_SIZE + i]
                        );
                        hidden_errors++;
                        sample_errors++;
                    end
                end

                for (int i = 0; i < OUTPUT_SIZE; i++) begin
                    if (
                        dut.output_mem[i] !==
                        expected_output_all[sample*OUTPUT_SIZE + i]
                    ) begin
                        $display(
                            "Sample %0d output[%0d] mismatch: RTL=%0d expected=%0d",
                            sample, i, dut.output_mem[i],
                            expected_output_all[sample*OUTPUT_SIZE + i]
                        );
                        output_errors++;
                        sample_errors++;
                    end
                end
            end

            total_cycles += cycle_count;
            total_errors += sample_errors;

            if (sample_errors == 0) begin
                $display(
                    "Sample %0d: PASS (prediction=%0d, cycles=%0d)",
                    sample, prediction, cycle_count
                );
            end else begin
                $display("Sample %0d: FAIL (%0d errors)", sample, sample_errors);
            end
        end

        $display("");
        $display("Samples tested:    %0d", NUM_TESTS);
        $display("Prediction errors: %0d", prediction_errors);
        $display("Hidden errors:     %0d", hidden_errors);
        $display("Output errors:     %0d", output_errors);
        $display("Total errors:      %0d", total_errors);
        $display("Average cycles:    %0d", total_cycles / NUM_TESTS);

        if (total_errors == 0) begin
            $display("OVERALL RESULT: PASS");
            $finish;
        end else begin
            $fatal(1, "OVERALL RESULT: FAIL");
        end
    end

endmodule
